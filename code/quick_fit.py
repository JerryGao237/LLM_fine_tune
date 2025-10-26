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
from datasets import load_dataset, Dataset
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
import glob
from pathlib import Path
from datasets import load_dataset, Dataset

# ---------------------------- 数据加载与校验 ----------------------------

def _is_local_path(p: str) -> bool:
    try:
        return Path(p).expanduser().exists()
    except Exception:
        return False

def _infer_builder_and_files(p: str):
    """根据本地路径推断 datasets builder 与文件列表"""
    path = Path(p).expanduser()
    if path.is_file():
        ext = path.suffix.lower()
        if ext in [".json", ".jsonl"]:
            return "json", str(path)
        if ext in [".parquet"]:
            return "parquet", str(path)
        raise ValueError(f"不支持的本地数据文件类型：{ext}")
    if path.is_dir():
        # 目录下自动找常见数据文件
        files = []
        files += glob.glob(str(path / "*.json"))
        files += glob.glob(str(path / "*.jsonl"))
        files += glob.glob(str(path / "*.parquet"))
        if not files:
            raise FileNotFoundError(f"目录 {path} 下未发现 json/jsonl/parquet 数据文件")
        # 优先 json/jsonl
        json_like = [f for f in files if Path(f).suffix.lower() in (".json", ".jsonl")]
        if json_like:
            return "json", json_like
        parquet_like = [f for f in files if Path(f).suffix.lower() == ".parquet"]
        return "parquet", parquet_like
    raise FileNotFoundError(f"本地路径不存在：{path}")

def load_and_cast_dataset(method: str, dataset_name: str, split: str = "train") -> Dataset:
    """加载数据并进行基本列校验；本地 json/jsonl/parquet 走相应 builder。"""
    if _is_local_path(dataset_name):
        builder, files = _infer_builder_and_files(dataset_name)
        ds = load_dataset(builder, data_files=files, split=split)
    else:
        # HF Hub 或自定义脚本名
        ds = load_dataset(dataset_name, split=split)

    cols = set(ds.column_names)
    m = method.lower()

    if m == "sft":
        if not ({"messages", "text", "prompt"} & cols or ({"prompt", "completion"} <= cols)):
            raise ValueError("SFT 需要 messages/text/prompt 之一；或 prompt+completion")
    elif m in {"dpo", "orpo"}:
        assert {"chosen", "rejected"} <= cols, f"{method} 需要列: chosen, rejected"
    elif m == "kto":
        assert {"prompt", "completion", "label"} <= cols, "KTO 需要列: prompt, completion, label"
    elif m == "grpo":
        assert "prompt" in cols, "GRPO 需要列: prompt"
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

    if m == "sft":
        args = SFTConfig(
            output_dir=output_dir,
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=500 if eval_ds else None,
            save_steps=500,
            logging_steps=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=300,
            learning_rate=2e-4,
            warmup_steps=0,
            save_total_limit=3,
            load_best_model_at_end=True if eval_ds else False,
            metric_for_best_model="eval_loss" if eval_ds else None,
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
        )
        trainer = KTOTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif m == "grpo":
        # 你可以将 reward 函数替换为更实际的打分器（规则/奖励模型/外部评测）
        def reward_fn(completions, **kwargs):
            """
            completions: List[List[{"role": "assistant", "content": str}, ...]]
            返回与 completions 同长度的浮点奖励列表。
            这里给一个玩具示例：奖励"不同字符多"的输出。
            """
            def to_text(comp):
                if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                    return comp[0].get("content", "")
                return str(comp)
            return [float(len(set(to_text(c)))) for c in completions]

        args = GRPOConfig(
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
