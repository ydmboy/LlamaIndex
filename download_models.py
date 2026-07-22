"""download_models.py — 下载 embedding 模型到 models/ 目录。

用法：
  python download_models.py                      # 下载默认模型 bge-m3
  python download_models.py --model bge-m3       # 下载 bge-m3（约 4.25 GB）
  python download_models.py --model bge-small-zh # 下载 bge-small-zh-v1.5（约 95 MB）
  python download_models.py --model all-minilm   # 下载 all-MiniLM-L6-v2（约 95 MB）
  python download_models.py --list               # 列出所有可用模型
"""

import argparse
import os
import sys
from pathlib import Path

# 可用模型清单
MODELS = {
    "bge-m3": {
        "repo_id": "BAAI/bge-m3",
        "local_dir": "models/bge-m3",
        "description": "BAAI bge-m3 多语言 embedding（推荐，约 4.25 GB）",
        "size_gb": 4.25,
    },
    "bge-small-zh": {
        "repo_id": "BAAI/bge-small-zh-v1.5",
        "local_dir": "models/bge-small-zh-v1.5",
        "description": "BAAI bge-small-zh-v1.5 中文 embedding（轻量，约 95 MB）",
        "size_gb": 0.095,
    },
    "all-minilm": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "local_dir": "models/all-MiniLM-L6-v2",
        "description": "all-MiniLM-L6-v2 英文 embedding（轻量，约 95 MB）",
        "size_gb": 0.095,
    },
}


def list_models():
    """列出所有可用模型"""
    print("\n可用 embedding 模型：")
    print("-" * 70)
    for key, info in MODELS.items():
        exists = Path(info["local_dir"]).exists() and any(
            Path(info["local_dir"]).glob("*.safetensors")
        )
        status = "✅ 已下载" if exists else "⬜ 未下载"
        print(f"  {key:<15} {info['description']}")
        print(f"  {'':15} HuggingFace: {info['repo_id']}")
        print(f"  {'':15} 本地路径:   {info['local_dir']}")
        print(f"  {'':15} 大小:       {info['size_gb']} GB | 状态: {status}")
        print()


def download_model(model_key: str):
    """下载指定模型"""
    if model_key not in MODELS:
        print(f"错误：未知模型 '{model_key}'")
        print(f"可用模型：{', '.join(MODELS.keys())}")
        sys.exit(1)

    info = MODELS[model_key]
    local_dir = Path(info["local_dir"])

    # 检查是否已存在
    if local_dir.exists() and any(local_dir.glob("*.safetensors")):
        print(f"✅ 模型已存在：{local_dir}")
        print(f"   跳过下载")
        return

    print(f"开始下载：{info['description']}")
    print(f"  HuggingFace: {info['repo_id']}")
    print(f"  保存到:      {local_dir}")
    print(f"  预估大小:    {info['size_gb']} GB")
    print()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("错误：未安装 huggingface-hub，请先运行：pip install huggingface-hub")
        sys.exit(1)

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=info["repo_id"],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print()
        print(f"✅ 下载完成：{local_dir}")
        print(f"   使用方式：$env:EMBED_MODEL=\"{local_dir}\"")
    except Exception as e:
        print(f"❌ 下载失败：{e}")
        print(f"   可手动下载：https://huggingface.co/{info['repo_id']}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="下载 embedding 模型")
    parser.add_argument(
        "--model",
        default="bge-m3",
        help=f"模型名称（默认 bge-m3，可选：{', '.join(MODELS.keys())}）",
    )
    parser.add_argument("--list", action="store_true", help="列出所有可用模型")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    download_model(args.model)


if __name__ == "__main__":
    main()
