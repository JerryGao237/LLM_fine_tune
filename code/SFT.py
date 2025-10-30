# -*- coding: utf-8 -*-
"""
train_and_chat.py

统一调用一套接口，覆盖以下微调方法：
- SFT（监督微调）
- DPO（Direct Preference Optimization）
- ORPO（Odds Ratio Preference Optimization）
- KTO（Kahneman-Tversky Optimization）
- GRPO（Group Relative Policy Optimization）
并支持一键 QLoRA（4bit 量化 + LoRA）训练。

依赖（建议）:
    pip install -U "trl>=0.8" transformers datasets accelerate peft bitsandbytes

示例：
    # SFT + QLoRA
    python train_and_chat.py --method sft \
      --model Qwen/Qwen2-0.5B-Instruct \
      --dataset trl-lib/Capybara \
      --out runs/qwen-sft-qlora \
      --qlora

    # DPO + QLoRA
    python train_and_chat.py --method dpo \
      --model Qwen/Qwen2-0.5B-Instruct \
      --dataset trl-lib/ultrafeedback_binarized \
      --out runs/qwen-dpo-qlora \
      --qlora

    # ORPO + QLoRA
    python train_and_chat.py --method orpo \
      --model Qwen/Qwen2-0.5B-Instruct \
      --dataset trl-lib/ultrafeedback_binarized \
      --out runs/qwen-orpo-qlora \
      --qlora

    # KTO + QLoRA
    python train_and_chat.py --method kto \
      --model Qwen/Qwen2-0.5B-Instruct \
      --dataset trl-lib/kto-mix-14k \
      --out runs/qwen-kto-qlora \
      --qlora

    # GRPO + QLoRA（在线采样 + 自定义 reward）
    python train_and_chat.py --method grpo \
      --model Qwen/Qwen2-0.5B-Instruct \
      --dataset trl-lib/ultrafeedback-prompt \
      --out runs/qwen-grpo-qlora \
      --qlora

对话推理（统一 messages 输入）
    python train_and_chat.py --method sft --model Qwen/Qwen2-0.5B-Instruct \
      --dataset trl-lib/Capybara --out runs/qwen-sft \
      --chat '[{"role":"user","content":"给我 3 条学习微积分的建议"}]'

数据列期望（与 TRL 文档对齐）:
    SFT:
        {"messages": [...]} 或 {"prompt": str, "completion": str} 或 {"text": str}
    DPO / ORPO:
        {"prompt": str (可选), "chosen": str, "rejected": str}
    KTO:
        {"prompt": str, "completion": str, "label": int|bool}
    GRPO:
        {"prompt": str}
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Optional

import torch
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import (
    SFTTrainer, SFTConfig,
    DPOTrainer, DPOConfig,
    ORPOTrainer, ORPOConfig,
    KTOTrainer, KTOConfig,
    GRPOTrainer, GRPOConfig,
)


# ---------------------------- 数据加载与校验 ----------------------------

def load_and_cast_dataset(method: str, dataset_name: str, split: str = "train") -> Dataset:
    """加载数据并进行基本列校验，不做重写转换，尽量沿用 TRL 内置处理。
    """
    print("methods:", method)
    print("DatasetName:", dataset_name)
    if dataset_name.endswith(".parquet"):
        # 如果是本地的 parquet 文件，直接加载
        ds = Dataset.from_parquet(dataset_name)
    else:
        # 如果是数据集名称（如HuggingFace数据集），则使用 load_dataset
        ds = load_dataset(dataset_name, split=split)
    cols = set(ds.column_names)
    m = method.lower()

    if m == "sft":
        # 对于 ultrafeedback_binarized，使用 chosen 列作为 SFT 数据
        if "chosen" in cols and "messages" in cols:
            # ultrafeedback_binarized 的特殊处理：chosen 列就是对话数据
            # 重命名 chosen -> messages（如果需要）或直接使用
            # 实际上 messages 列已经存在，但可能需要验证格式
            pass
        elif not ({"messages", "text", "prompt"} & cols):
            raise ValueError("SFT 需要 messages/text/prompt 之一；也可用 prompt+completion")
    elif m in {"dpo", "orpo"}:
        assert {"chosen", "rejected"} <= cols, f"{method} 需要列: chosen, rejected"
        # prompt 可选
    elif m == "kto":
        # 数据处理
        chosen_ds = ds.map(
            lambda x: {"prompt": [x["chosen"][0]], "completion": [x["chosen"][1]], "label": True},
            remove_columns=[c for c in cols if c not in ["prompt", "completion", "label"]],
        )
        rejected_ds = ds.map(
            lambda x: {"prompt": [x["rejected"][0]], "completion": [x["rejected"][1]], "label": False},
            remove_columns=[c for c in cols if c not in ["prompt", "completion", "label"]],
        )
        ds = concatenate_datasets([chosen_ds, rejected_ds]).shuffle(seed=42)
        cols = set(ds.column_names)
        assert {"prompt", "completion", "label"} <= cols, (
            "KTO 需要列: prompt, completion, label"
        )
    elif m == "grpo":
        assert "prompt" in cols, "GRPO 需要列: prompt"
        # 对于 GRPO，通常需要 chosen 和 rejected 对的 pair 来计算奖励
        # 确保 chosen 和 rejected 列存在并且处理正确
        # 处理 GRPO 数据：格式化为适合奖励对齐的方法
        # 可能需要选择或删除一些无效样本，这取决于你的具体需求
        # ds = ds.select(range(5))
        ds = ds.map(
            lambda x: {
                "chosen": x["chosen"][1]["content"],
                "rejected": x["rejected"][1]["content"],
                "prompt": x["prompt"],
                "score_chosen": x["score_chosen"] if "score_chosen" in x else 0,
                "score_rejected": x["score_rejected"] if "score_rejected" in x else 0,
            },
            remove_columns=[c for c in cols if c not in ["chosen", "rejected", "prompt", "score_chosen", "score_rejected"]],
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return ds


# ---------------------------- LoRA 目标模块小助手 ----------------------------

def guess_lora_targets(model) -> Optional[List[str]]:
    """尽力猜测常见模型的 LoRA 注入位置；若失败则返回 None。
    你也可以通过 CLI --lora_targets 显式指定。
    """
    name = type(model).__name__.lower()
    common = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if any(k in name for k in ["llama", "mistral", "qwen", "phi", "gemma", "yi"]):
        return common
    # Qwen2 系的注意力命名有时是 c_attn / c_proj / W_pack
    for cand in ("W_pack", "c_attn", "c_proj"):
        for n, _ in model.named_modules():
            if cand in n:
                return common + [cand]
    return None


# ---------------------------- 训练分发器 ----------------------------

def build_model_and_tokenizer(
        model_name_or_path: str,
        qlora: bool,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, Optional[BitsAndBytesConfig]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # k-bit 训练准备
        model = prepare_model_for_kbit_training(model)
        try:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", trust_remote_code=True
        )

    return model, tokenizer, bnb_config

def get_precision_config():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False}
        else:
            if hasattr(torch.cuda, 'is_fp16_supported') and torch.cuda.is_fp16_supported():
                return {"bf16": False, "fp16": True}
            else:
                return {"bf16": False, "fp16": False}
    else:
        return {"bf16": False, "fp16": False}

def train(
        method: str,
        model_name_or_path: str,
        dataset_name: str,
        output_dir: str,
        split: str = "train",
        eval_split: Optional[str] = None,
        peft: bool = False,
        qlora: bool = False,
        lora_targets: Optional[str] = "",
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
):
    ds = load_and_cast_dataset(method, dataset_name, split)
    eval_ds = None

    if eval_split:
        eval_ds = load_and_cast_dataset(method, dataset_name, eval_split)

    # 数据采样：用于快速验证
    if max_train_samples and max_train_samples > 0:
        original_size = len(ds)
        ds = ds.select(range(min(max_train_samples, len(ds))))
        print(f"[INFO] 训练集采样: {original_size} -> {len(ds)} 样本")

    if eval_ds and max_eval_samples and max_eval_samples > 0:
        original_size = len(eval_ds)
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))
        print(f"[INFO] 验证集采样: {original_size} -> {len(eval_ds)} 样本")

    model, tokenizer, _ = build_model_and_tokenizer(model_name_or_path, qlora)

    # LoRA/QLoRA 配置
    peft_config = None
    if qlora or peft:
        targets = [t for t in (lora_targets or "").split(",") if t.strip()] or guess_lora_targets(model)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )

    m = method.lower()
    precision_config = get_precision_config()
    if m == "sft":
        print("Choose SFTTrainer")
        args = SFTConfig(
            output_dir=output_dir,
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=500 if eval_ds else None,
            save_steps=500,
            logging_steps=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            warmup_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True if eval_ds else False,
            metric_for_best_model="eval_loss" if eval_ds else None,
            **precision_config
        )
        # 对于 ultrafeedback_binarized，直接使用 chosen 列作为对话数据
        # 删除可能导致冲突的 messages 列
        if "chosen" in ds.column_names and "messages" in ds.column_names:
            ds = ds.remove_columns(["messages", "rejected", "prompt"])
            ds = ds.rename_column("chosen", "messages")
            if eval_ds:
                eval_ds = eval_ds.remove_columns(["messages", "rejected", "prompt"])
                eval_ds = eval_ds.rename_column("chosen", "messages")
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif m == "dpo":
        print("Choose DPOTrainer")
        args = DPOConfig(
            output_dir=output_dir,
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=500 if eval_ds else None,
            save_steps=500,
            logging_steps=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            learning_rate=5e-5,
            warmup_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True if eval_ds else False,
            metric_for_best_model="eval_loss" if eval_ds else None,
            **precision_config
        )
        # 对于 ultrafeedback_binarized，删除多余的 messages 列
        # DPOTrainer 只需要 prompt, chosen, rejected
        if "messages" in ds.column_names:
            ds = ds.remove_columns(["messages"])
            if eval_ds and "messages" in eval_ds.column_names:
                eval_ds = eval_ds.remove_columns(["messages"])

        trainer = DPOTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif m == "orpo":
        print("Choose ORPOTrainer")
        args = ORPOConfig(
            output_dir=output_dir,
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=500 if eval_ds else None,
            save_steps=500,
            logging_steps=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            learning_rate=8e-6,
            warmup_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True if eval_ds else False,
            metric_for_best_model="eval_loss" if eval_ds else None,
            **precision_config
        )
        # 对于 ultrafeedback_binarized，删除多余的 messages 列
        # ORPOTrainer 只需要 prompt, chosen, rejected
        if "messages" in ds.column_names:
            ds = ds.remove_columns(["messages"])
            if eval_ds and "messages" in eval_ds.column_names:
                eval_ds = eval_ds.remove_columns(["messages"])

        trainer = ORPOTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif m == "kto":
        print("Choose KTOTrainer")
        args = KTOConfig(
            output_dir=output_dir,
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=500 if eval_ds else None,
            save_steps=500,
            logging_steps=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            learning_rate=5e-5,
            warmup_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True if eval_ds else False,
            metric_for_best_model="eval_loss" if eval_ds else None,
            **precision_config
        )

        def to_text(x):
            if x is None:
                return ""
            if isinstance(x, list):
                # 如果总是单元素列表，也可以写 return x[0]
                return "\n".join(str(i) for i in x)
            return str(x)

        def normalize(example):
            example["prompt"] = to_text(example.get("prompt"))
            example["completion"] = to_text(example.get("completion"))
            # label 已经是 bool，无需改；若不是则转成 bool(example["label"])
            return example

        train_dataset = ds.map(normalize)
        train_dataset = train_dataset.cast(Features({
            "prompt": Value("string"),
            "completion": Value("string"),
            "label": Value("bool"),
        }))
     
        trainer = KTOTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif m == "grpo":
        # 你可以将 reward 函数替换为更实际的打分器（规则/奖励模型/外部评测）
        def reward_fn(prompts,completions, **kwargs):
            """
            completions: List[List[{"role": "assistant", "content": str}, ...]]
            返回与 completions 同长度的浮点奖励列表。
            这里给一个玩具示例：奖励"不同字符多"的输出。
            """
            chosen_list = kwargs.get("chosen")
            rejected_list = kwargs.get("rejected")
            score_chosen_list = kwargs.get("score_chosen")
            score_rejected_list = kwargs.get("score_rejected")

            # 初始化 tokenizer（可以使用适当的预训练模型）
            # tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer = AutoTokenizer.from_pretrained("./model/base")
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained("./model/base", output_hidden_states=True)
            rewards = []

            # 遍历每个样本
            for prompt, completion, chosen, rejected, score_chosen, score_rejected in zip(prompts, completions, chosen_list, rejected_list, score_chosen_list, score_rejected_list):
                # 使用 tokenizer 将文本转换为模型的输入 ID
                completion_input = tokenizer(completion, return_tensors="pt", padding=True, truncation=True,max_length=512)
                chosen_input = tokenizer(chosen, return_tensors="pt", padding=True, truncation=True,max_length=512)
                rejected_input = tokenizer(rejected, return_tensors="pt", padding=True, truncation=True,max_length=512)
                with torch.no_grad():  # 禁用梯度计算
                    completion_outputs = model(**completion_input)
                    chosen_outputs = model(**chosen_input)
                    rejected_outputs = model(**rejected_input)
                    # 获取最后一层的隐藏状态，并对其进行平均
                    completion_embedding = completion_outputs.hidden_states[-1].mean(dim=1)
                    chosen_embedding = chosen_outputs.hidden_states[-1].mean(dim=1)
                    rejected_embedding = rejected_outputs.hidden_states[-1].mean(dim=1)
         
                def cosine_similarity(x, y):
                    x = x.float()
                    y = y.float()
                    return torch.nn.functional.cosine_similarity(x, y, dim=-1)
                # 计算生成文本与选择项和拒绝项的相似度
                similarity_chosen = cosine_similarity(completion_embedding, chosen_embedding)
                similarity_rejected = cosine_similarity(completion_embedding, rejected_embedding)

                # 基于相似度和得分计算奖励
                reward = 2*(score_chosen * similarity_chosen - score_rejected * similarity_rejected)
                rewards.append(reward)
            return rewards

        args = GRPOConfig(
            output_dir=output_dir,
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=500 if eval_ds else None,
            save_steps=500,
            logging_steps=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            learning_rate=2e-6,
            warmup_steps=100,
            save_total_limit=3,
            # generation_batch_size=8,  # 修改这个值，确保它是 num_generations 的倍数
            num_generations=8,  # 修改这
            bf16=False,  # 禁用 bf16
            fp16=False,  # 禁用 fp16
        )
        trainer = GRPOTrainer(
            model=model,  # 传已加载的模型对象（可含 QLoRA/LoRA）
            reward_funcs=reward_fn,
            args=args,
            train_dataset=ds,
            eval_dataset=eval_ds,
            peft_config=peft_config,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    trainer.train()
    trainer.save_model(output_dir)

    # 保存训练状态（包含 loss、评估指标等）
    trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    try:
        tokenizer.save_pretrained(output_dir)
    except Exception:
        pass

    print(f"[OK] Trained with {method.upper()} -> {output_dir}")


# ---------------------------- 统一对话推理 ----------------------------

def chat(
        model_or_dir: str,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
) -> str:
    """
    统一的对话接口：
      - 输入：messages = [{"role":"system|user|assistant","content":"..."}, ...]
      - 输出：字符串（assistant 生成）
    优先使用 Transformers 的结构化 messages；失败时回退到 chat template。
    """
    tok = AutoTokenizer.from_pretrained(model_or_dir, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    textgen = pipeline(
        "text-generation", model=model_or_dir, tokenizer=tok, device_map="auto"
    )

    # 首选：直接喂 messages
    try:
        out = textgen(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            return_full_text=False,
        )
        gen = out[0]["generated_text"]
        if isinstance(gen, list):
            # 若返回结构化对话，取最后一条 assistant
            for turn in reversed(gen):
                if isinstance(turn, dict) and turn.get("role") == "assistant":
                    return turn.get("content", "")
            return json.dumps(gen, ensure_ascii=False)
        return str(gen)
    except Exception:
        # 回退：chat template 或极简模板
        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            sys_msgs = "\n".join(
                [m["content"] for m in messages if m.get("role") == "system"]
            )
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            prompt = (f"{sys_msgs}\nUser: {last_user}\nAssistant: ").strip()

        out = textgen(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            return_full_text=False,
        )
        return out[0]["generated_text"]


# ---------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified training + chat for TRL methods + QLoRA")
    ap.add_argument("--method", required=True, choices=["sft", "dpo", "orpo", "kto", "grpo"], help="fine-tuning method")
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--dataset", required=True, help="HF dataset name or local data script")
    ap.add_argument("--split", default="train", help="dataset split")
    ap.add_argument("--eval_split", default="", help="evaluation dataset split (optional)")
    ap.add_argument("--out", required=True, help="output dir")

    ap.add_argument("--peft", action="store_true", help="enable LoRA (no quant)")
    ap.add_argument("--qlora", action="store_true", help="enable QLoRA (4-bit + LoRA)")
    ap.add_argument("--lora_targets", type=str, default="", help="comma-separated target_modules; empty = auto-guess")

    # 数据采样参数
    ap.add_argument("--max_train_samples", type=int, default=0, help="max training samples (0 = use all)")
    ap.add_argument("--max_eval_samples", type=int, default=0, help="max evaluation samples (0 = use all)")

    ap.add_argument("--chat", type=str, default="", help="JSON array messages to run a quick chat after training")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)

    args = ap.parse_args()

    train(
        method=args.method,
        model_name_or_path=args.model,
        dataset_name=args.dataset,
        output_dir=args.out,
        split=args.split,
        eval_split=args.eval_split if args.eval_split else None,
        peft=args.peft,
        qlora=args.qlora,
        lora_targets=args.lora_targets,
        max_train_samples=args.max_train_samples if args.max_train_samples > 0 else None,
        max_eval_samples=args.max_eval_samples if args.max_eval_samples > 0 else None,
    )

    if args.chat:
        try:
            messages = json.loads(args.chat)
        except Exception as e:
            raise SystemExit(f"--chat 参数需要 JSON 数组：{e}")
        reply = chat(args.out, messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print("\n[CHAT OUTPUT]\n" + reply)


if __name__ == "__main__":
    main()
