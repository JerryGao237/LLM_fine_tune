#!/usr/bin/env python3
"""
统一评测脚本 - 自动评估 runs 目录下的所有训练结果，输出 JSON 格式

功能：
  1. 自动发现 runs 目录下的所有训练结果
  2. 计算所有评估指标（Loss、Perplexity、偏好准确率）
  3. 生成 JSON 格式的完整评估报告

用法:
    # 评估 runs 目录下的所有结果
    python unified_evaluation.py

    # 指定其他目录
    python unified_evaluation.py --runs_dir experiments/

    # 只评估特定方法
    python unified_evaluation.py --methods sft dpo

    # 跳过耗时的 Perplexity 计算
    python unified_evaluation.py --skip_perplexity
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class UnifiedEvaluator:
    def __init__(
        self,
        runs_dir: str = "runs",
        base_model: str = "Qwen/Qwen2-0.5B-Instruct",
        dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
        max_samples: int = 100,
        skip_perplexity: bool = False,
        max_generation_samples: int = 3,
    ):
        self.runs_dir = Path(runs_dir)
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.skip_perplexity = skip_perplexity
        self.max_generation_samples = max_generation_samples

    def discover_models(self) -> List[Path]:
        """自动发现 runs 目录下的所有训练结果"""
        print(f"\n{'='*80}")
        print(f"🔍 扫描目录: {self.runs_dir}")
        print(f"{'='*80}\n")

        if not self.runs_dir.exists():
            print(f"❌ 目录不存在: {self.runs_dir}")
            return []

        models = []
        for item in sorted(self.runs_dir.iterdir()):
            if item.is_dir():
                has_adapter = (item / "adapter_model.safetensors").exists()
                has_model = (item / "pytorch_model.bin").exists() or (item / "model.safetensors").exists()

                if has_adapter or has_model:
                    models.append(item)
                    model_type = "LoRA" if has_adapter else "Full"
                    print(f"  ✓ {item.name} ({model_type})")

        print(f"\n共发现 {len(models)} 个模型\n")
        return models

    def load_training_state(self, model_path: Path) -> Optional[Dict]:
        """加载训练状态"""
        state_file = model_path / "trainer_state.json"

        if not state_file.exists():
            checkpoints = sorted([d for d in model_path.iterdir()
                                if d.is_dir() and d.name.startswith("checkpoint-")])
            if checkpoints:
                state_file = checkpoints[-1] / "trainer_state.json"

        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return None

    def extract_training_metrics(self, state: Dict) -> Dict:
        """从训练状态中提取指标"""
        if not state:
            return {}

        log_history = state.get("log_history", [])

        # 训练 loss
        train_logs = [log for log in log_history if "loss" in log and "eval_loss" not in log]
        initial_train_loss = train_logs[0]["loss"] if train_logs else None
        final_train_loss = train_logs[-1]["loss"] if train_logs else None

        # 验证 loss
        eval_logs = [log for log in log_history if "eval_loss" in log]
        eval_losses = [log["eval_loss"] for log in eval_logs]
        best_eval_loss = min(eval_losses) if eval_losses else None
        final_eval_loss = eval_losses[-1] if eval_losses else None

        # 训练信息
        total_steps = state.get("global_step", 0)
        best_checkpoint = state.get("best_model_checkpoint", None)

        # 训练时间
        training_time_seconds = 0
        if log_history and "train_runtime" in log_history[-1]:
            training_time_seconds = log_history[-1]["train_runtime"]

        return {
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "best_eval_loss": best_eval_loss,
            "final_eval_loss": final_eval_loss,
            "total_steps": total_steps,
            "best_checkpoint": best_checkpoint,
            "training_time_seconds": training_time_seconds,
            "training_time_hours": training_time_seconds / 3600 if training_time_seconds else 0,
        }

    def calculate_perplexity(self, model, tokenizer, dataset, max_samples: int = 100) -> tuple:
        """计算困惑度"""
        total_loss = 0
        total_tokens = 0

        for i, sample in enumerate(tqdm(dataset, desc="  计算 Perplexity", total=min(max_samples, len(dataset)))):
            if i >= max_samples:
                break

            if "messages" in sample:
                messages = sample["messages"]
            elif "chosen" in sample:
                messages = sample["chosen"]
            else:
                continue

            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except:
                text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                num_tokens = inputs["input_ids"].numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = np.exp(avg_loss)

        return float(perplexity), float(avg_loss)

    def calculate_preference_accuracy(self, model, tokenizer, dataset, max_samples: int = 100) -> Dict:
        """计算偏好准确率"""
        correct = 0
        total = 0
        chosen_losses = []
        rejected_losses = []

        for sample in tqdm(dataset, desc="  计算偏好准确率", total=min(max_samples, len(dataset))):
            if total >= max_samples:
                break

            chosen = sample.get("chosen", [])
            rejected = sample.get("rejected", [])

            if not chosen or not rejected:
                continue

            # 计算 chosen 的 loss
            try:
                chosen_text = tokenizer.apply_chat_template(
                    chosen, tokenize=False, add_generation_prompt=False
                )
            except:
                chosen_text = "\n".join([f"{m['role']}: {m['content']}" for m in chosen])

            chosen_inputs = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=512)
            chosen_inputs = {k: v.to(model.device) for k, v in chosen_inputs.items()}

            with torch.no_grad():
                chosen_output = model(**chosen_inputs, labels=chosen_inputs["input_ids"])
                chosen_loss = chosen_output.loss.item()

            # 计算 rejected 的 loss
            try:
                rejected_text = tokenizer.apply_chat_template(
                    rejected, tokenize=False, add_generation_prompt=False
                )
            except:
                rejected_text = "\n".join([f"{m['role']}: {m['content']}" for m in rejected])

            rejected_inputs = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=512)
            rejected_inputs = {k: v.to(model.device) for k, v in rejected_inputs.items()}

            with torch.no_grad():
                rejected_output = model(**rejected_inputs, labels=rejected_inputs["input_ids"])
                rejected_loss = rejected_output.loss.item()

            if chosen_loss < rejected_loss:
                correct += 1

            total += 1
            chosen_losses.append(chosen_loss)
            rejected_losses.append(rejected_loss)

        if total == 0:
            return {}

        return {
            "preference_accuracy": float(correct / total),
            "preference_correct": correct,
            "preference_total": total,
            "avg_chosen_loss": float(np.mean(chosen_losses)),
            "avg_rejected_loss": float(np.mean(rejected_losses)),
            "loss_margin": float(np.mean(rejected_losses) - np.mean(chosen_losses)),
        }

    def generate_samples(self, model, tokenizer, prompts: List[str], max_new_tokens: int = 100) -> List[Dict]:
        """生成样例回复"""
        results = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]

            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                text = f"User: {prompt}\nAssistant:"

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Assistant:" in generated:
                reply = generated.split("Assistant:")[-1].strip()
            else:
                reply = generated[len(text):].strip()

            results.append({"prompt": prompt, "response": reply})

        return results

    def evaluate_model(self, model_path: Path) -> Dict:
        """评估单个模型"""
        print(f"\n{'='*80}")
        print(f"📊 评估: {model_path.name}")
        print(f"{'='*80}\n")

        results = {
            "model_name": model_path.name,
            "model_path": str(model_path),
        }

        # 1. 加载训练状态
        print("📈 加载训练指标...")
        state = self.load_training_state(model_path)
        if state:
            train_metrics = self.extract_training_metrics(state)
            results.update(train_metrics)
            print(f"  ✓ 最佳验证 Loss: {train_metrics.get('best_eval_loss', 'N/A')}")
        else:
            print("  ⚠️  未找到训练状态文件")

        # 2. 加载模型
        if not self.skip_perplexity:
            try:
                print("\n🤖 加载模型...")
                tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                adapter_config = model_path / "adapter_config.json"
                if adapter_config.exists():
                    base = AutoModelForCausalLM.from_pretrained(
                        self.base_model,
                        device_map="auto",
                        torch_dtype=torch.bfloat16
                    )
                    model = PeftModel.from_pretrained(base, str(model_path))
                    print("  ✓ LoRA 模型")
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(model_path),
                        device_map="auto",
                        torch_dtype=torch.bfloat16
                    )
                    print("  ✓ 完整模型")

                model.eval()

                # 3. 加载测试数据集
                print("\n📚 加载测试集...")
                try:
                    test_ds = load_dataset(self.dataset_name, split="test_sft")
                    print("  ✓ test_sft")
                except:
                    try:
                        test_ds = load_dataset(self.dataset_name, split="test_prefs")
                        print("  ✓ test_prefs")
                    except:
                        print("  ⚠️  无法加载测试集")
                        test_ds = None

                # 4. 计算 Perplexity
                if test_ds:
                    print(f"\n🧮 计算困惑度 ({self.max_samples} 样本)...")
                    perplexity, avg_loss = self.calculate_perplexity(
                        model, tokenizer, test_ds, self.max_samples
                    )
                    results["perplexity"] = perplexity
                    results["test_loss"] = avg_loss
                    print(f"  ✓ Perplexity: {perplexity:.2f}")

                # 5. 计算偏好准确率（所有模型都计算，作为对比基线）
                try:
                    pref_ds = load_dataset(self.dataset_name, split="test_prefs")
                    print(f"\n🎯 计算偏好准确率 ({self.max_samples} 样本)...")
                    pref_metrics = self.calculate_preference_accuracy(
                        model, tokenizer, pref_ds, self.max_samples
                    )
                    results.update(pref_metrics)
                    if pref_metrics:
                        print(f"  ✓ 准确率: {pref_metrics['preference_accuracy']:.2%}")
                except Exception as e:
                    print(f"  ⚠️  跳过偏好准确率: {e}")

                # 6. 生成样例
                print(f"\n💬 生成样例 ({self.max_generation_samples} 个)...")
                test_prompts = [
                    "给我3条学习编程的建议",
                    "如何保持健康的生活方式？",
                    "解释一下什么是机器学习",
                ][:self.max_generation_samples]

                samples = self.generate_samples(model, tokenizer, test_prompts, max_new_tokens=100)
                results["generation_samples"] = samples
                print(f"  ✓ 完成")

                # 清理显存
                del model
                if 'base' in locals():
                    del base
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ❌ 模型评估失败: {e}")

        return results

    def run(self, filter_methods: Optional[List[str]] = None) -> Dict:
        """运行完整评估流程"""
        # 1. 发现模型
        models = self.discover_models()

        if not models:
            print("❌ 未发现任何训练结果")
            return {}

        # 2. 过滤模型
        if filter_methods:
            models = [m for m in models if any(method in m.name.lower() for method in filter_methods)]
            print(f"过滤后: {len(models)} 个模型\n")

        # 3. 评估每个模型
        all_results = {}
        for model_path in models:
            try:
                results = self.evaluate_model(model_path)
                all_results[model_path.name] = results
            except Exception as e:
                print(f"❌ 评估 {model_path.name} 失败: {e}\n")

        # 4. 保存结果
        print(f"\n{'='*80}")
        print("💾 保存结果")
        print(f"{'='*80}\n")

        output_file = "evaluation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"✓ 结果已保存: {output_file}")

        # 5. 打印摘要
        print(f"\n{'='*80}")
        print("📊 评估摘要")
        print(f"{'='*80}\n")

        print(f"{'模型':<30} {'验证Loss':<12} {'困惑度':<12} {'偏好准确率':<12}")
        print("-" * 66)

        for name, results in sorted(all_results.items()):
            eval_loss = results.get("best_eval_loss")
            ppl = results.get("perplexity")
            pref_acc = results.get("preference_accuracy")

            eval_loss_str = f"{eval_loss:.4f}" if eval_loss else "N/A"
            ppl_str = f"{ppl:.2f}" if ppl else "N/A"
            pref_acc_str = f"{pref_acc*100:.2f}%" if pref_acc else "N/A"

            print(f"{name:<30} {eval_loss_str:<12} {ppl_str:<12} {pref_acc_str:<12}")

        print(f"\n✅ 评估完成！结果保存在: {output_file}\n")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="统一评测脚本 - 输出 JSON 格式结果")
    parser.add_argument("--runs_dir", type=str, default="runs", help="训练结果目录")
    parser.add_argument("--methods", nargs="+", help="只评估指定方法 (如: sft dpo)")
    parser.add_argument("--max_samples", type=int, default=100, help="每个指标使用的最大样本数")
    parser.add_argument("--max_generation_samples", type=int, default=3, help="生成样例的数量")
    parser.add_argument("--skip_perplexity", action="store_true", help="跳过 Perplexity 计算")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="基座模型")

    args = parser.parse_args()

    evaluator = UnifiedEvaluator(
        runs_dir=args.runs_dir,
        base_model=args.base_model,
        max_samples=args.max_samples,
        skip_perplexity=args.skip_perplexity,
        max_generation_samples=args.max_generation_samples,
    )

    evaluator.run(filter_methods=args.methods)


if __name__ == "__main__":
    main()
