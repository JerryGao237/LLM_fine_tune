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


# ---------------------------- 数据加载与校验 ----------------------------

def load_and_cast_dataset(method: str, dataset_name: str, split: str = "train") -> Dataset:
    """加载数据并进行基本列校验，不做重写转换，尽量沿用 TRL 内置处理。
    """
    ds = load_dataset(dataset_name, split=split)
    cols = set(ds.column_names)
    m = method.lower()

    if m == "sft":
        assert {"messages", "text", "prompt"} & cols, (
            "SFT 需要 messages/text/prompt 之一；也可用 prompt+completion"
        )
    elif m in {"dpo", "orpo"}:
        assert {"chosen", "rejected"} <= cols, f"{method} 需要列: chosen, rejected"
        # prompt 可选
    elif m == "kto":
        assert {"prompt", "completion", "label"} <= cols, (
            "KTO 需要列: prompt, completion, label"
        )
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
    peft: bool = False,
    qlora: bool = False,
    lora_targets: Optional[str] = "",
):
    ds = load_and_cast_dataset(method, dataset_name, split)

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
        args = SFTConfig(output_dir=output_dir)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif m == "dpo":
        args = DPOConfig(output_dir=output_dir)
        trainer = DPOTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif m == "orpo":
        args = ORPOConfig(output_dir=output_dir)
        trainer = ORPOTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif m == "kto":
        args = KTOConfig(output_dir=output_dir)
        trainer = KTOTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif m == "grpo":
        # 你可以将 reward 函数替换为更实际的打分器（规则/奖励模型/外部评测）
        def reward_fn(completions, **kwargs):
            """
            completions: List[List[{"role": "assistant", "content": str}, ...]]
            返回与 completions 同长度的浮点奖励列表。
            这里给一个玩具示例：奖励“不同字符多”的输出。
            """
            def to_text(comp):
                if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                    return comp[0].get("content", "")
                return str(comp)
            return [float(len(set(to_text(c)))) for c in completions]

        args = GRPOConfig(output_dir=output_dir)
        trainer = GRPOTrainer(
            model=model,  # 传已加载的模型对象（可含 QLoRA/LoRA）
            reward_funcs=reward_fn,
            args=args,
            train_dataset=ds,
            peft_config=peft_config,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    trainer.train()
    trainer.save_model(output_dir)
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
    ap.add_argument("--out", required=True, help="output dir")

    ap.add_argument("--peft", action="store_true", help="enable LoRA (no quant)")
    ap.add_argument("--qlora", action="store_true", help="enable QLoRA (4-bit + LoRA)")
    ap.add_argument("--lora_targets", type=str, default="", help="comma-separated target_modules; empty = auto-guess")

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
        peft=args.peft,
        qlora=args.qlora,
        lora_targets=args.lora_targets,
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
