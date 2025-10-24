# 大语言模型微调框架

这是一个统一的大语言模型微调框架，支持多种微调方法（SFT、DPO、ORPO、KTO、GRPO），并提供完整的评测工具。

## 📋 目录

- [项目简介](#项目简介)
- [支持的方法](#支持的方法)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [评测工具](#评测工具)
- [输出结构](#输出结构)
- [常见问题](#常见问题)

## 项目简介

本项目实现了一个统一的大语言模型微调框架，具有以下特点：

- **多方法支持**：支持 SFT、DPO、ORPO、KTO、GRPO 五种微调方法
- **高效训练**：使用 QLoRA（4-bit 量化 + LoRA）实现参数高效微调
- **自动评测**：提供统一评测脚本，自动评估所有训练结果
- **完整指标**：包括 Loss、Perplexity、偏好准确率等多维度评估
- **灵活配置**：支持数据采样、验证评估、自定义参数等

## 支持的方法

| 方法 | 类型 | 数据集要求 | 学习率 | 说明 |
|------|------|-----------|--------|------|
| SFT | 监督微调 | 单条回复（messages） | 2e-4 | 基础微调方法 |
| DPO | 偏好对齐 | 偏好对（chosen/rejected） | 5e-5 | 直接偏好优化 |
| ORPO | 偏好对齐 | 偏好对（chosen/rejected） | 8e-6 | 奇异比偏好优化 |
| KTO | 偏好对齐 | 单条标注（label） | 5e-5 | Kahneman-Tversky 优化 |
| GRPO | 强化学习 | prompt 列表 | 5e-5 | 群体相对策略优化 |

## 环境配置

### 必需依赖

```bash
pip install torch transformers datasets accelerate peft bitsandbytes trl
```

### 推荐环境

- Python 3.8+
- CUDA 11.8+
- GPU 显存 ≥ 16GB（使用 QLoRA 可在更小显存上运行）

## 快速开始

### 1. SFT 监督微调

```bash
python code/SFT.py \
  --method sft \
  --model_name Qwen/Qwen2-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --split train_sft \
  --eval_split test_sft \
  --output_dir runs/sft_qwen_0.5b \
  --max_train_samples 1000 \
  --max_eval_samples 100
```

### 2. DPO 偏好优化

```bash
python code/SFT.py \
  --method dpo \
  --model_name Qwen/Qwen2-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --split train_prefs \
  --eval_split test_prefs \
  --output_dir runs/dpo_qwen_0.5b \
  --max_train_samples 1000 \
  --max_eval_samples 100
```

### 3. 运行评测

```bash
# 评测所有模型
python code/unified_evaluation.py

# 只评测特定方法
python code/unified_evaluation.py --methods sft dpo

# 跳过耗时的 Perplexity 计算
python code/unified_evaluation.py --skip_perplexity
```

## 详细使用

### SFT.py 命令行参数

#### 基础参数

- `--method`：微调方法，可选 `sft`、`dpo`、`orpo`、`kto`、`grpo`（必需）
- `--model_name`：基座模型名称，如 `Qwen/Qwen2-0.5B-Instruct`（必需）
- `--dataset`：数据集名称，如 `HuggingFaceH4/ultrafeedback_binarized`（必需）
- `--output_dir`：输出目录，建议使用 `runs/{method}_{model}` 格式（必需）

#### 数据参数

- `--split`：训练数据集分割，默认 `train`
  - SFT 使用：`train_sft`
  - DPO/ORPO 使用：`train_prefs`
  - GRPO 使用：`train_gen`
- `--eval_split`：验证数据集分割，如 `test_sft`、`test_prefs`
- `--max_train_samples`：训练样本数量上限（可选，用于快速实验）
- `--max_eval_samples`：验证样本数量上限（可选）

#### 训练参数

- `--num_train_epochs`：训练轮数，默认 `3`
- `--per_device_train_batch_size`：单设备训练批次大小，默认 `4`
- `--gradient_accumulation_steps`：梯度累积步数，默认 `4`
- `--learning_rate`：学习率（自动根据方法设置）
- `--use_qlora`：启用 QLoRA 4-bit 量化，默认开启

#### LoRA 参数

- `--lora_r`：LoRA 秩，默认 `16`
- `--lora_alpha`：LoRA alpha，默认 `32`
- `--lora_dropout`：LoRA dropout，默认 `0.05`

### 完整训练示例

#### SFT 监督微调

```bash
python code/SFT.py \
  --method sft \
  --model_name Qwen/Qwen2-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --split train_sft \
  --eval_split test_sft \
  --output_dir runs/sft_qwen_0.5b \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_samples 1000 \
  --max_eval_samples 100
```

#### DPO 直接偏好优化

```bash
python code/SFT.py \
  --method dpo \
  --model_name Qwen/Qwen2-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --split train_prefs \
  --eval_split test_prefs \
  --output_dir runs/dpo_qwen_0.5b \
  --num_train_epochs 3 \
  --max_train_samples 1000 \
  --max_eval_samples 100
```

#### ORPO 奇异比偏好优化

```bash
python code/SFT.py \
  --method orpo \
  --model_name Qwen/Qwen2-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --split train_prefs \
  --eval_split test_prefs \
  --output_dir runs/orpo_qwen_0.5b \
  --num_train_epochs 3 \
  --max_train_samples 1000 \
  --max_eval_samples 100
```

#### KTO Kahneman-Tversky 优化

```bash
python code/SFT.py \
  --method kto \
  --model_name Qwen/Qwen2-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --split train_prefs \
  --eval_split test_prefs \
  --output_dir runs/kto_qwen_0.5b \
  --num_train_epochs 3 \
  --max_train_samples 1000 \
  --max_eval_samples 100
```

#### GRPO 群体相对策略优化

```bash
python code/SFT.py \
  --method grpo \
  --model_name Qwen/Qwen2-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --split train_gen \
  --eval_split test_gen \
  --output_dir runs/grpo_qwen_0.5b \
  --num_train_epochs 3 \
  --max_train_samples 1000 \
  --max_eval_samples 100
```

## 评测工具

### unified_evaluation.py

统一评测脚本，自动发现并评估 `runs` 目录下的所有训练结果。

#### 命令行参数

- `--runs_dir`：训练结果目录，默认 `runs`
- `--methods`：只评测指定方法，如 `--methods sft dpo`
- `--max_samples`：每个指标使用的最大样本数，默认 `100`
- `--max_generation_samples`：生成样例的数量，默认 `3`
- `--skip_perplexity`：跳过 Perplexity 计算（加快评测速度）
- `--base_model`：基座模型名称，默认 `Qwen/Qwen2-0.5B-Instruct`

#### 使用示例

```bash
# 评测所有模型（完整评估）
python code/unified_evaluation.py

# 只评测 SFT 和 DPO
python code/unified_evaluation.py --methods sft dpo

# 快速评估（跳过 Perplexity）
python code/unified_evaluation.py --skip_perplexity

# 使用更多样本评估
python code/unified_evaluation.py --max_samples 500

# 指定其他目录
python code/unified_evaluation.py --runs_dir experiments/
```

#### 评测指标

评测脚本会自动计算以下指标：

**训练指标**（从 trainer_state.json 提取）：
- `initial_train_loss`：初始训练损失
- `final_train_loss`：最终训练损失
- `best_eval_loss`：最佳验证损失
- `final_eval_loss`：最终验证损失
- `total_steps`：总训练步数
- `training_time_seconds`：训练时间（秒）
- `training_time_hours`：训练时间（小时）

**测试指标**（在测试集上计算）：
- `perplexity`：困惑度（越低越好）
- `test_loss`：测试集损失
- `preference_accuracy`：偏好准确率（chosen loss < rejected loss 的比例）
- `preference_correct`：偏好正确的样本数
- `preference_total`：偏好评估的总样本数
- `avg_chosen_loss`：chosen 回复的平均损失
- `avg_rejected_loss`：rejected 回复的平均损失
- `loss_margin`：损失差值（rejected - chosen，越大越好）

**生成样例**：
- `generation_samples`：模型生成的样例回复（包含 prompt 和 response）

#### JSON 输出格式

评测结果保存在 `evaluation_results.json`：

```json
{
  "sft_qwen_0.5b": {
    "model_name": "sft_qwen_0.5b",
    "model_path": "runs/sft_qwen_0.5b",
    "initial_train_loss": 2.1234,
    "final_train_loss": 1.2345,
    "best_eval_loss": 1.1234,
    "final_eval_loss": 1.1456,
    "total_steps": 750,
    "training_time_seconds": 3600,
    "training_time_hours": 1.0,
    "perplexity": 3.14,
    "test_loss": 1.1432,
    "preference_accuracy": 0.65,
    "preference_correct": 65,
    "preference_total": 100,
    "avg_chosen_loss": 1.05,
    "avg_rejected_loss": 1.35,
    "loss_margin": 0.30,
    "generation_samples": [
      {
        "prompt": "给我3条学习编程的建议",
        "response": "1. 每天坚持练习..."
      }
    ]
  },
  "dpo_qwen_0.5b": {
    ...
  }
}
```

## 输出结构

训练完成后，每个模型的输出目录结构如下：

```
runs/{method}_{model}/
├── adapter_config.json          # LoRA 配置（~1KB）
├── adapter_model.safetensors    # LoRA 权重（~5-10MB）
├── trainer_state.json           # 训练状态和指标（~50KB）
├── training_args.bin            # 训练参数（~10KB）
├── tokenizer_config.json        # Tokenizer 配置
├── special_tokens_map.json      # 特殊 token 映射
└── checkpoint-{step}/           # 检查点目录（如果启用）
    ├── adapter_model.safetensors
    └── trainer_state.json
```

详细说明请参考 [RUNS_STRUCTURE.md](RUNS_STRUCTURE.md)。

## 常见问题

### Q1: 数据集格式要求

**A:** 不同方法对数据集格式有不同要求：

- **SFT**：需要 `messages` 列，格式为对话列表
- **DPO/ORPO**：需要 `prompt`、`chosen`、`rejected` 列
- **KTO**：需要 `prompt`、`completion`、`label` 列
- **GRPO**：需要 `prompt` 列

推荐使用 `HuggingFaceH4/ultrafeedback_binarized` 数据集，已包含所需格式。

### Q2: 如何减少显存占用

**A:** 可以尝试以下方法：

1. 减小批次大小：`--per_device_train_batch_size 2`
2. 增加梯度累积：`--gradient_accumulation_steps 8`
3. 减小 LoRA 秩：`--lora_r 8`
4. 使用数据采样：`--max_train_samples 500`

### Q3: 训练时间过长

**A:** 可以尝试：

1. 使用数据采样快速验证：`--max_train_samples 1000`
2. 减少训练轮数：`--num_train_epochs 1`
3. 跳过验证评估：不指定 `--eval_split`
4. 增大批次大小（如果显存允许）

### Q4: 如何选择合适的方法

**A:** 方法选择建议：

- **SFT**：作为基线，适合所有场景
- **DPO**：有成对偏好数据时的首选，效果稳定
- **ORPO**：在 DPO 基础上改进，通常效果更好
- **KTO**：只有单条评分数据时使用
- **GRPO**：强化学习场景，适合需要在线反馈的任务

### Q5: 偏好准确率的含义

**A:** 偏好准确率衡量模型是否给 chosen（高质量）回复分配更低的损失。具体计算：

- 对于每个样本，分别计算 chosen 和 rejected 的损失
- 如果 `chosen_loss < rejected_loss`，则认为预测正确
- 准确率 = 正确样本数 / 总样本数

这个指标对所有模型都计算，作为基线对比。通常：
- **SFT**：50-60%（未经偏好对齐）
- **DPO/ORPO**：65-75%（经过偏好对齐）

### Q6: 如何使用训练好的模型

**A:** 训练完成后，可以这样加载模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    device_map="auto"
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, "runs/sft_qwen_0.5b")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# 生成回复
messages = [{"role": "user", "content": "你好"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 开发指南

### 添加新的微调方法

1. 在 `SFT.py` 的 `main()` 函数中添加新的 `elif` 分支
2. 从 TRL 导入相应的 Trainer
3. 配置训练参数和数据格式
4. 更新 README 和文档

### 添加新的评测指标

1. 在 `unified_evaluation.py` 的 `UnifiedEvaluator` 类中添加新方法
2. 在 `evaluate_model()` 中调用新方法
3. 更新 JSON 输出格式
4. 更新 README 的指标说明

### 自定义数据集

如果使用自定义数据集，需要确保：

1. 数据格式符合所选方法的要求
2. 使用 `code/check_dataset.py` 检查数据格式
3. 必要时修改 `SFT.py` 中的数据预处理逻辑

## 参考资料

- [TRL 文档](https://huggingface.co/docs/trl)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Qwen2 模型](https://huggingface.co/Qwen)
- [UltraFeedback 数据集](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

## 许可证

本项目仅供学习和研究使用。
