# -*- coding: utf-8 -*-
"""
LLM Fine-tune Demo (本地离线可运行)
- 🧪 推理展示：本地加载基座模型（model/ 目录或绝对路径），可叠加 runs/ 下的 LoRA 适配器推理
- 🛠️ 现场微调：粘贴小样例 JSON（SFT/DPO/ORPO/KTO），一键 LoRA/QLoRA 微调，输出到 runs/ 目录
- 📊 微调展示：读取 evaluation_results.json 展示排行榜，可调用 code/unified_evaluation.py 重跑

运行：
    pip install -r requirements.txt
    python main.py
"""
import os
import re
import json
import time
import glob
import subprocess
from typing import Dict, Any, List, Optional, Tuple

# ---- 强制本地离线，不会联网下载 ----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------- 全局默认路径（可在 UI 中覆盖） ----------
MODEL_DIR   = os.environ.get("DEMO_MODEL_DIR", "model")
RUNS_DIR    = os.environ.get("DEMO_RUNS_DIR", "runs")
EVAL_SCRIPT = os.environ.get("DEMO_EVAL_SCRIPT", "code/unified_evaluation.py")
EVAL_JSON   = os.environ.get("DEMO_EVAL_JSON", "results/evaluation_results.json")
DEFAULT_BASE_LOCAL = os.environ.get("DEMO_BASE_MODEL", "base")  # 表示使用 model/base

# ------------------------- 工具函数 -------------------------
def _fmt_exception(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)}"

def _resolve_local_model_path(user_value: str) -> str:
    """
    解析本地模型路径：
    - 若 user_value 是现存目录，直接使用；
    - 否则尝试 MODEL_DIR/user_value；
    - 否则抛错（不联网）。
    """
    v = (user_value or "").strip() or "base"
    if os.path.isdir(v):
        return v
    cand = os.path.join(MODEL_DIR, v)
    if os.path.isdir(cand):
        return cand
    raise FileNotFoundError(f"未找到本地模型目录：'{v}' 或 '{cand}'。请将模型放在 {MODEL_DIR}/ 子目录或提供绝对路径。")

def list_adapters(runs_dir: str = RUNS_DIR) -> List[Tuple[str, str]]:
    """
    返回 (显示标签, 路径)，寻找含 adapter_model.safetensors 的目录；
    先查 run 根目录；再从最高 checkpoint-* 往下找。
    """
    items: List[Tuple[str, str]] = []
    if not os.path.isdir(runs_dir):
        return items

    for run_name in sorted(os.listdir(runs_dir)):
        full = os.path.join(runs_dir, run_name)
        if not os.path.isdir(full):
            continue

        root_adapter = os.path.join(full, "adapter_model.safetensors")
        if os.path.exists(root_adapter):
            items.append((run_name, full))
            continue

        ckpts = sorted(
            (d for d in glob.glob(os.path.join(full, "checkpoint-*")) if os.path.isdir(d)),
            key=lambda p: int(re.findall(r"checkpoint-(\d+)", p)[0]) if re.findall(r"checkpoint-(\d+)", p) else -1,
            reverse=True
        )
        for ck in ckpts:
            ck_adapter = os.path.join(ck, "adapter_model.safetensors")
            if os.path.exists(ck_adapter):
                items.append((f"{run_name}/{os.path.basename(ck)}", ck))
                break
    return items

_MODEL_SLOT: Dict[str, Any] = {"key": None, "tok": None, "model": None}

def unload_model():
    global _MODEL_SLOT
    _MODEL_SLOT = {"key": None, "tok": None, "model": None}
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

def _maybe_quant_args(quant_mode: str) -> Dict[str, Any]:
    q = (quant_mode or "none").lower()
    if q == "8bit" or q == "4bit":
        # Windows 下未装 bitsandbytes 会失败；交由 load_model 捕获并返回报错
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            raise RuntimeError("选择了 8bit/4bit，但环境未安装 bitsandbytes 或系统不支持。请改回 'none'。")
        if q == "8bit":
            return dict(load_in_8bit=True, device_map="auto")
        return dict(load_in_4bit=True, device_map="auto")
    # 默认 float16（有 GPU）
    return dict(device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else None)

def load_model(base_model_hint: str, adapter_dir: Optional[str], quant_mode: str = "none"):
    """
    严格本地加载：AutoTokenizer / AutoModelForCausalLM（local_files_only=True），可叠加 LoRA 适配器。
    使用一个单槽缓存，避免频繁重复加载。
    """
    global _MODEL_SLOT
    base_path = _resolve_local_model_path(base_model_hint)
    key = (base_path, adapter_dir or "", quant_mode)
    if _MODEL_SLOT["key"] == key and _MODEL_SLOT["model"] is not None:
        return _MODEL_SLOT["tok"], _MODEL_SLOT["model"]

    unload_model()
    print('base_path:', base_path)
    quant_kwargs = _maybe_quant_args(quant_mode)
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        **quant_kwargs
    )

    base.eval()
    model = PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base

    _MODEL_SLOT = {"key": key, "tok": tok, "model": model}
    return tok, model

def apply_chat_template(tok: AutoTokenizer, user_text: str) -> Dict[str, Any]:
    """
    将用户输入封装为聊天模板（若可用），否则原样编码。
    """
    try:
        messages = [{"role": "user", "content": user_text}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt")
    except Exception:
        return tok(user_text, return_tensors="pt")

# ------------------------- 推理展示逻辑 -------------------------
def sync_runs_dropdown(runs_dir: str) -> Tuple[List[str], List[str]]:
    items = list_adapters(runs_dir)
    labels = ["(仅基座模型)"] + [lbl for (lbl, _) in items]
    paths = [""] + [pth for (_, pth) in items]
    return labels, paths

def infer_once(base_model_hint: str, adapter_choice_path: str, user_text: str,
               max_new_tokens: int, temperature: float, top_p: float,
               quant_mode: str, seed: int) -> Tuple[str, str]:
    if not (user_text or "").strip():
        return "", "请输入内容。"
    torch.manual_seed(int(seed))
    start = time.time()
    try:
        tok, model = load_model(base_model_hint, adapter_choice_path or None, quant_mode=quant_mode)
        # ---- 这里统计 LoRA 层与活动适配器信息 ----
        active = getattr(model, "active_adapter", None)
        try:
            import peft
            from peft.tuners.lora import LoraLayer
            lora_layers = sum(1 for _, m in model.named_modules() if isinstance(m, LoraLayer))
        except Exception:
            lora_layers = -1

        inputs = apply_chat_template(tok, user_text)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_kwargs = {}
        if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = tok.eos_token_id

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                **gen_kwargs
            )
        # 只解码新生成部分（避免把提示词也解码回去，看起来“更像一样”）
        input_len = inputs["input_ids"].shape[1]
        text = tok.decode(outputs[0][input_len:], skip_special_tokens=True)

        info = f"{time.time() - start:.3f}s | adapter={adapter_choice_path or '(none)'} | active={active} | lora_layers={lora_layers}"
        return text, info
    except Exception as e:
        return "", f"推理失败：{_fmt_exception(e)}"


# ------------------------- 评测展示逻辑 -------------------------
def try_read_eval_json(path: str) -> Tuple[Optional[Dict[str, Any]], str]:
    if not os.path.exists(path):
        return None, f"未找到 {path}。可先点击按钮运行评测，或在项目根目录放置该文件。"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, f"已加载 {path}（{len(data)} 个条目）。"
    except Exception as e:
        return None, f"读取 {path} 失败：{_fmt_exception(e)}"

def eval_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, v in data.items():
        rows.append({
            "model_name": name,
            "model_path": v.get("model_path"),
            "perplexity": v.get("perplexity"),
            "test_loss": v.get("test_loss"),
            "training_time_hours": v.get("training_time_hours"),
            "total_steps": v.get("total_steps"),
        })
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["perplexity", "test_loss"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[False]*len(sort_cols), na_position="last")
    return df

def pick_generation_samples(data: Dict[str, Any], model_name: str) -> List[Dict[str, str]]:
    item = data.get(model_name) or {}
    return item.get("generation_samples", []) or []

def run_evaluation(methods: str, runs_dir: str, base_model_hint: str, skip_perplexity: bool, max_samples: int) -> str:
    if not os.path.exists(EVAL_SCRIPT):
        return f"未找到评测脚本：{EVAL_SCRIPT}"
    try:
        base_path = _resolve_local_model_path(base_model_hint)
    except Exception as e:
        return _fmt_exception(e)
    cmd = ["python", EVAL_SCRIPT, "--runs_dir", runs_dir, "--base_model", base_path, "--max_samples", str(int(max_samples))]
    parts = [p.strip() for p in (methods or "").split(",") if p.strip()]
    if parts:
        cmd += ["--methods"] + parts
    skip_perplexity = False
    if skip_perplexity:
        cmd += ["--skip_perplexity"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = []
        out.append("Command: " + " ".join(cmd))
        out.append("--- STDOUT ---\n" + (proc.stdout or ""))
        out.append("--- STDERR ---\n" + (proc.stderr or ""))
        return "\n".join(out)
    except Exception as e:
        return f"运行失败：{_fmt_exception(e)}"


def format_sft_text(messages: List[Dict[str, str]]) -> str:
    """将SFT格式的消息列表转换为文本"""
    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        text_parts.append(f"{role}: {content}")
    return "\n".join(text_parts)

def save_to_parquet(data_text: str, output_path: str):
    """
    将不同格式的文本数据保存到parquet文件

    Args:
        data_text: 包含JSON格式数据的文本字符串
        output_path: 输出的parquet文件路径
    """
    # 解析文本为JSON对象列表
    try:
        # 尝试解析为JSON数组
        data_list = json.loads(data_text)
        if not isinstance(data_list, list):
            data_list = [data_list]
    except json.JSONDecodeError:
        # 如果不是有效的JSON，尝试逐行解析
        lines = data_text.strip().split('\n')
        data_list = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data_list.append(item)
                except json.JSONDecodeError:
                    print(f"跳过无效的JSON行: {line}")
                    continue

    if not data_list:
        raise ValueError("没有找到有效的JSON数据")

    # 处理每个数据项
    processed_data = []

    for item in data_list:
        if not isinstance(item, dict):
            print(f"跳过非字典项: {item}")
            continue

        # 识别数据类型并标准化格式
        if "messages" in item:
            # SFT格式: {"messages": [{"role":"user", "content":"..."}, {"role":"assistant", "content":"..."}]}
            processed_item = {
                "type": "sft",
                "messages": item["messages"],
                "text": format_sft_text(item["messages"])
            }

        elif "chosen" in item and "rejected" in item:
            # DPO/ORPO格式: {"prompt":"...", "chosen":"...", "rejected":"..."}
            processed_item = {
                "type": "dpo",
                "prompt": item.get("prompt", ""),
                "chosen": item.get("chosen", ""),
                "rejected": item.get("rejected", ""),
                "text": f"Prompt: {item.get('prompt', '')}\nChosen: {item.get('chosen', '')}\nRejected: {item.get('rejected', '')}"
            }

        elif "completion" in item and "label" in item:
            # KTO格式: {"prompt":"...", "completion":"...", "label": 1或0}
            processed_item = {
                "type": "kto",
                "prompt": item.get("prompt", ""),
                "completion": item.get("completion", ""),
                "label": item.get("label", 0),
                "text": f"Prompt: {item.get('prompt', '')}\nCompletion: {item.get('completion', '')}\nLabel: {item.get('label', 0)}"
            }
        else:
            print(f"无法识别的数据格式: {item}")
            continue

        processed_data.append(processed_item)

    # 创建DataFrame
    df = pd.DataFrame(processed_data)

    # 保存为parquet文件
    df.to_parquet(output_path, index=False, engine='pyarrow')

# ------------------------- UI -------------------------
custom_css = """
.gradio-container {
    margin: 0 auto !important;
    max-width: 100% !important;
    width: 100% !important;
}
.container {
    max-width: 100% !important;
}
.main-title {
    text-align: center;
    margin-bottom: 20px;
}
.step-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px;
    border-radius: 10px;
    color: white;
    margin-bottom: 15px;
}
.info-box {
    background: #f0f7ff;
    padding: 12px;
    border-left: 4px solid #2196F3;
    border-radius: 5px;
    margin: 10px 0;
}
"""

with gr.Blocks(title="🤖 LLM 微调系统", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <div class="main-title">
    <h1>🤖 大语言模型微调系统</h1>
    <p style="font-size: 16px; color: #666;">本地离线运行 | 支持多种微调方法 | 完整评测体系</p>
    </div>
    """)
    
    gr.Markdown("""
    <div class="info-box">
    💡 <b>使用说明：</b>本系统完全离线运行，基座模型从 <code>model/</code> 目录加载，微调权重从 <code>runs/</code> 目录选择。
    </div>
    """)

    # ========== 推理展示 ==========
    with gr.Tab("💬 对话生成"):
        gr.Markdown("### 📝 步骤 1：配置模型")

        with gr.Row():
            with gr.Column():
                base_model_hint = gr.Textbox(
                    value=DEFAULT_BASE_LOCAL,
                    label="🎯 基座模型路径",
                    placeholder="输入 model 子目录名或完整路径",
                    info="例如：Qwen2-0.5B-Instruct 或绝对路径",
                    lines=1
                )
            with gr.Column():
                quant_mode = gr.Radio(
                    choices=["none", "8bit", "4bit"],
                    value="none",
                    label="💾 量化模式",
                    info="量化可减少显存占用"
                )

        gr.Markdown("### 📦 步骤 2：选择微调权重（可选）")

        runs_dir = gr.Textbox(
            value=RUNS_DIR,
            label="📁 权重目录",
            lines=1
        )
        refresh_btn = gr.Button("🔄 刷新权重列表", variant="secondary", size="sm")

        adapter_path = gr.State(value="")
        adapter_label = gr.Dropdown(
            choices=["(仅基座模型)"],
            value="(仅基座模型)",
            label="🎨 选择 LoRA 适配器",
            info="留空则使用基座模型"
        )
        refresh_info = gr.Markdown("")

        gr.Markdown("### ⚙️ 步骤 3：设置生成参数")

        with gr.Accordion("🔧 高级参数设置", open=False):
            with gr.Row():
                with gr.Column():
                    max_new_tokens = gr.Slider(
                        minimum=16, maximum=2048, step=16, value=256,
                        label="📏 最大生成长度",
                        info="生成的最大token数"
                    )
                    temperature = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.1, value=0.7,
                        label="🌡️ 温度 (Temperature)",
                        info="控制生成的随机性，越高越随机"
                    )
                with gr.Column():
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, step=0.05, value=0.9,
                        label="🎲 Top-p",
                        info="核采样参数，控制多样性"
                    )
                    seed = gr.Number(
                        value=42, precision=0,
                        label="🌱 随机种子",
                        info="固定种子可复现结果"
                    )

        gr.Markdown("### 💭 步骤 4：输入问题并生成")

        user_text = gr.Textbox(
            label="✍️ 输入内容（单轮对话）",
            lines=5,
            placeholder="请输入您的问题，例如：\n• 给我3条学习编程的建议\n• 解释一下什么是深度学习\n• 写一首关于春天的诗",
            info="支持多行输入"
        )

        go = gr.Button("🚀 开始生成", variant="primary", size="lg")

        gr.Markdown("### 📤 生成结果")

        with gr.Row():
            with gr.Column(scale=2):
                out_text = gr.Textbox(
                    label="💬 模型回复",
                    lines=12,
                    show_copy_button=True
                )
            with gr.Column(scale=1):
                out_lat = gr.Textbox(
                    label="⏱️ 生成信息",
                    lines=12
                )

        def _refresh(runs_dir_str: str):
            labels, paths = sync_runs_dropdown(runs_dir_str)
            # 默认选择第一项（仅基座）
            return gr.update(choices=labels, value=labels[0]), "" , f"已找到 {max(0, len(labels)-1)} 个适配器。"

        refresh_btn.click(_refresh, inputs=[runs_dir], outputs=[adapter_label, adapter_path, refresh_info])

        def _choose_adapter(label: str, runs_dir_str: str):
            labels, paths = sync_runs_dropdown(runs_dir_str)
            mapping = {lbl: pth for lbl, pth in zip(labels, paths)}
            return mapping.get(label, "")

        adapter_label.change(_choose_adapter, inputs=[adapter_label, runs_dir], outputs=[adapter_path])

        go.click(
            infer_once,
            inputs=[base_model_hint, adapter_path, user_text, max_new_tokens, temperature, top_p, quant_mode, seed],
            outputs=[out_text, out_lat]
        )

    # ========== 现场微调 ==========
    with gr.Tab("🎓 模型训练"):
        gr.Markdown("""
        <div class="info-box">
        🎯 <b>快速微调：</b>粘贴少量样本数据，一键启动 LoRA/QLoRA 微调，生成新的模型适配器。
        </div>
        """)
        
        gr.Markdown("### 🎯 步骤 1：选择微调方法")
        
        method2 = gr.Radio(
            choices=["sft","dpo","orpo","kto", 'grpo'],
            value="sft", 
            label="🔬 微调算法",
            info="SFT=监督微调 | DPO=直接偏好优化 | ORPO=奇偶比优化 | KTO=标签优化 | GRPO=群相对策略优化"
        )
        
        gr.Markdown("### 🎯 步骤 2：配置训练参数")
        
        with gr.Row():
            with gr.Column(scale=2):
                base_local = gr.Textbox(
                    value=DEFAULT_BASE_LOCAL, 
                    label="🎯 基座模型路径",
                    placeholder="例如：Qwen2-0.5B-Instruct",
                    info="从 model/ 目录选择或输入完整路径"
                )
            with gr.Column(scale=2):
                out_name = gr.Textbox(
                    value="runs/quick_sft", 
                    label="💾 输出目录",
                    placeholder="例如：runs/my_model",
                    info="训练结果保存位置（建议在 runs/ 下）"
                )

        with gr.Row():
            use_qlora = gr.Checkbox(
                value=True,
                label="⚡ 启用 QLoRA (4-bit量化)",
                info="大幅减少显存占用，推荐开启"
            )
        
        gr.Markdown("### 📝 步骤 3：准备训练数据")
        gr.Markdown("""
        <div style="background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107;">
        💡 <b>数据格式提示：</b>
        <ul style="margin: 5px 0;">
        <li><b>SFT：</b> 对话格式 {"messages": [{"role":"user", "content":"..."}, {"role":"assistant", "content":"..."}]}</li>
        <li><b>DPO/ORPO/GRPO：</b> 偏好对 {"prompt":"...", "chosen":"...", "rejected":"..."}</li>
        <li><b>KTO：</b> 标注数据 {"prompt":"...", "completion":"...", "label": 1或0}</li>
        </ul>
        </div>
        """)
        
        data_json_txt = gr.Textbox(
            label="📋 训练数据 (JSON格式)",
            lines=12,
            value='[{"messages":[{"role":"user","content":"给我3条学习编程的建议"},{"role":"assistant","content":"1. 坚持练习\\n2. 阅读源码\\n3. 多做项目"}]}]',
            placeholder="粘贴 JSON 格式的训练数据...",
            info="支持粘贴或直接编辑"
        )
        
        btn_fit = gr.Button("⚡ 开始训练", variant="primary", size="lg")
        
        gr.Markdown("### 📊 训练日志")
        
        fit_log = gr.Textbox(
            label="📝 实时训练日志", 
            lines=20, 
            show_copy_button=True,
            placeholder="训练日志将在此处显示..."
        )
        
        btn_refresh_after = gr.Button(
            "🔄 训练完成后刷新对话生成页的模型列表", 
            variant="secondary"
        )

        def _sample_fill(m):
            if m=="sft":
                return '[{"messages":[{"role":"user","content":"给我3条学习编程的建议"},{"role":"assistant","content":"1. 坚持练习\\n2. 阅读源码\\n3. 多做项目"}]}]'
            if m in ("dpo","orpo"):
                return '[{"prompt":"写一段自我介绍","chosen":"大家好，我是…（清晰有条理）","rejected":"嗨…（含糊其辞）"}]'
            if m=="kto":
                return '[{"chosen":["给我一个学习计划","周一到周五…"],"rejected":["给我一个学习计划","随便学习…"]}]'
            if m=="grpo":
                return '''[{"prompt": "写一段自我介绍","chosen": [{"role": "user", "content": "写一段自我介绍"},{"role": "assistant", "content": "大家好，我是…（清晰有条理）"}],
                        "rejected": [{"role": "user", "content": "写一段自我介绍"}, {"role": "assistant", "content": "嗨…（含糊其辞）"}]}]'''
            return "[]"

        method2.change(_sample_fill, inputs=[method2], outputs=[data_json_txt])

        def _run_quick_fit(method, base_hint, out_dir, qlora_v, data_txt):
            print('Inputs: ', method, base_hint, out_dir, qlora_v, data_txt)

            import json, os, tempfile, subprocess
            try:
                base_path = _resolve_local_model_path(base_hint)
            except Exception as e:
                return f"解析本地模型失败：{e}"

            # --- 解析并写入临时 JSON（确保 Windows 不锁文件） ---
            try:
                data = json.loads(data_txt)
                if not isinstance(data, list):
                    return "data_json 必须是 JSON 列表（list[dict]）"
                fd, data_path = tempfile.mkstemp(suffix=".json")
                os.close(fd)  # 关键：立刻关闭句柄，避免 Windows 文件锁
                with open(data_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as e:
                return f"解析/写入训练数据失败：{e}"

            if not os.path.exists(out_dir):
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception as e:
                    return f"创建输出目录失败：{e}"
            data_path = f'{out_dir}/training_data.parquet'
            save_to_parquet(data_txt, data_path)

            # --- 组装命令并调用后端 ---
            cmd = [
                "python", "code/SFT.py",
                "--method", method,
                "--model", base_path,
                "--dataset", data_path,   # 传本地 JSON 路径
                "--out", out_dir
            ]
            if qlora_v:
                cmd.append("--qlora")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                out = []
                out.append("Command: " + " ".join(cmd))
                out.append("--- STDOUT ---\n" + (proc.stdout or ""))
                out.append("--- STDERR ---\n" + (proc.stderr or ""))
                return "\n".join(out)
            except Exception as e:
                return f"运行失败：{type(e).__name__}: {e}"
            finally:
                try:
                    os.remove(data_path)
                except Exception:
                    pass

        btn_fit.click(
            _run_quick_fit,
            inputs=[method2, base_local, out_name, use_qlora, data_json_txt],
            outputs=[fit_log]
        )

        # 刷新推理页 Dropdown
        btn_refresh_after.click(_refresh, inputs=[runs_dir], outputs=[adapter_label, adapter_path, refresh_info])

    # ========== 微调展示（评测） ==========
    with gr.Tab("📈 模型评测"):
        gr.Markdown("""
        <div class="info-box">
        📊 <b>全面评测：</b>自动评估所有训练模型，提供困惑度、偏好准确率等多维度指标。
        </div>
        """)
        
        gr.Markdown("### ⚙️ 步骤 1：配置评测参数")
        
        with gr.Row():
            with gr.Column(scale=2):
                methods = gr.Textbox(
                    value="", 
                    label="🔍 筛选方法（可选）",
                    placeholder="例如：sft,dpo （留空评测全部）",
                    info="用逗号分隔多个方法名"
                )
            with gr.Column(scale=2):
                runs_dir2 = gr.Textbox(
                    value=RUNS_DIR, 
                    label="📁 模型目录",
                    info="包含训练模型的目录"
                )
        
        with gr.Row():
            with gr.Column():
                base_model2 = gr.Textbox(
                    value=DEFAULT_BASE_LOCAL, 
                    label="🎯 基座模型路径",
                    info="用于评测的基础模型"
                )
            with gr.Column():
                max_samples = gr.Slider(
                    minimum=20, maximum=1000, value=100, step=20,
                    label="📊 评测样本数",
                    info="每个指标使用的测试样本数量"
                )
            with gr.Column():
                skip_pp = gr.Checkbox(
                    value=True,
                    label="⚡ 快速模式",
                    info="跳过困惑度计算以加快速度"
                )
        
        run_eval_btn = gr.Button("🚀 开始评测", variant="primary", size="lg")
        
        gr.Markdown("### 📋 评测日志")
        
        cmd_log = gr.Textbox(
            label="📝 评测执行日志", 
            lines=15, 
            show_copy_button=True,
            placeholder="评测日志将在此处显示..."
        )
        
        reload_btn = gr.Button("🔄 重新加载评测结果", variant="secondary")
        status = gr.Markdown("")

        gr.Markdown("### 🏆 模型排行榜")
        
        leaderboard = gr.Dataframe(
            label="📊 性能对比表（按偏好准确率和损失差值排序）", 
            interactive=False,
            wrap=True
        )
        
        gr.Markdown("### 🔍 生成样例查看")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_pick = gr.Dropdown(
                    choices=[], 
                    label="🎯 选择模型",
                    info="查看该模型的生成样例"
                )
            with gr.Column(scale=2):
                samples_box = gr.JSON(
                    label="💬 生成样例展示"
                )

        def _run_eval(methods_str, runs_dir_str, base_model_str, skip_pp_bool, max_samples_val):
            log = run_evaluation(methods_str, runs_dir_str, base_model_str, bool(skip_pp_bool), int(max_samples_val))
            return log

        run_eval_btn.click(_run_eval, inputs=[methods, runs_dir2, base_model2, skip_pp, max_samples], outputs=[cmd_log])

        def _reload_json():
            data, msg = try_read_eval_json(EVAL_JSON)
            if data is None:
                return gr.update(value=[]), gr.update(choices=[]), "[]", msg
            df = eval_to_dataframe(data)
            return df, gr.update(choices=list(data.keys()), value=(list(data.keys())[0] if data else None)), "[]", msg

        reload_btn.click(_reload_json, outputs=[leaderboard, model_pick, samples_box, status])

        def _pick_samples(model_name: str):
            data, _ = try_read_eval_json(EVAL_JSON)
            if not data or not model_name:
                return "[]"
            samples = pick_generation_samples(data, model_name)
            return json.dumps(samples, ensure_ascii=False, indent=2)

        model_pick.change(_pick_samples, inputs=[model_pick], outputs=[samples_box])

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    demo.launch(server_name="0.0.0.0", server_port=7860)
