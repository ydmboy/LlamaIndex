"""
交互式 RAG 问答：DeepSeek LLM + HuggingFace embedding + IngestionPipeline 哈希去重
数据源：Obsidian 笔记仓库 / 任意文档目录（.md/.txt 文件）

用法：
    $env:DEEPSEEK_API_KEY="你的key"
    .venv\Scripts\python.exe run_vector_demo.py

首次运行会读取数据源目录下所有 .md 文件，通过 IngestionPipeline 进行哈希去重、
切块、向量化，构建索引并持久化到 ./storage；之后启动若检测到 storage 存在则直接
加载，实现"秒启动"。在 REPL 中输入问题即时获取回答，输入 exit/quit/退出 停止。

去重原理（两道防线）：
  1) SimHash 近似转载拦截：摄入前对正文归一化（NFKC+去标点空白）算 64 位内容指纹，
     与库内指纹汉明距离 <= 3 即判为转载跳过——同一篇文章被不同报纸转载、
     或同一内容存成不同文件（file_path 不同导致文档哈希必然不同）都能拦住。
  2) IngestionPipeline + DocstoreStrategy.DUPLICATES_ONLY：对每个文档计算 SHA256
     内容哈希，哈希已存在于 docstore → 跳过（不切块、不向量化、不存储）；
     哈希不存在 → 切块、向量化、存入向量库 + 记录哈希。完全不依赖文档格式。
"""
import logging
import sys
import os
import re
import json
import math
import hashlib
import unicodedata
import numpy as np
import torch
from pathlib import Path

# 日志：INFO 级别可看到 DeepSeek HTTP 请求；想更安静改成 WARNING
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    Settings,
)
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

# LLM 提供商可选（按需导入，避免无用依赖报错）
try:
    from llama_index.llms.openai_like import OpenAILike
except ImportError:
    OpenAILike = None
try:
    from llama_index.llms.ollama import Ollama
except ImportError:
    Ollama = None

# ---- 检索参数配置（retrieval_config.yaml）----
RETRIEVAL_CONFIG_FILE = "retrieval_config.yaml"


def _load_retrieval_config() -> dict:
    """从 retrieval_config.yaml 加载检索参数。文件不存在时用默认值，保证程序可独立运行。"""
    default = {
        "vector": {"similarity_top_k": 5, "response_mode": "tree_summarize"},
        "fulltext": {"top_k": 20, "bm25_k1": 1.5, "bm25_b": 0.75},
        "chunk": {"chunk_size": 512, "chunk_overlap": 50},
        "highlight": {"top_n": 2, "threshold": 0.5},
        "ingest": {"batch_size": 1000, "auto_continue_timeout": 10},
        "dedup": {"simhash_enabled": True, "simhash_threshold": 3, "simhash_min_chars": 8},
    }
    config_path = Path(RETRIEVAL_CONFIG_FILE)
    if not config_path.exists():
        return default
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # 按分区合并：用户配置覆盖默认值
        merged = {}
        for section, sec_default in default.items():
            merged[section] = {**sec_default, **(user_cfg.get(section) or {})}
        return merged
    except Exception as e:
        print(f"[警告] 加载检索配置失败 {RETRIEVAL_CONFIG_FILE}: {e}，使用默认值")
        return default


_RETR_CFG = _load_retrieval_config()


def _make_query_engine(index):
    """统一的 query_engine 创建，参数来自 retrieval_config.yaml（CLI/Web 共用）。"""
    return index.as_query_engine(
        response_mode=_RETR_CFG["vector"]["response_mode"],
        similarity_top_k=_RETR_CFG["vector"]["similarity_top_k"],
    )


# ---- 补丁：流式持久化，避免大库写盘时内存翻倍 ----
# SimpleKVStore.persist 原版是 f.write(json.dumps(...))——先在内存里拼出完整
# JSON 字符串（大库可达数十 GB）再一次性写盘，持久化阶段极易 MemoryError。
# 改为 json.dump(obj, f) 流式写入：写盘字节完全相同，但不再产生巨型字符串。
import json as _json

import fsspec as _fsspec
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore as _SimpleKVStore


def _streaming_kvstore_persist(self, persist_path, fs=None):
    fs = fs or _fsspec.filesystem("file")
    dirpath = os.path.dirname(persist_path)
    if not fs.exists(dirpath):
        fs.makedirs(dirpath)
    with fs.open(persist_path, "w", encoding="utf-8") as f:
        _json.dump(self._collections_mappings, f)


_SimpleKVStore.persist = _streaming_kvstore_persist

# ---- 配置（均可用环境变量覆盖，便于跨机器部署）----
DATA_DIR = os.environ.get("DATA_DIR", r"D:\wiki\beijing_daily\2026-06-30")
STORAGE_DIR = os.environ.get("STORAGE_DIR", "./storage")
INDEX_ID = "vector_index"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")  # 空=启动时交互选择，或设为本地路径/HF ID
CHUNK_SIZE = _RETR_CFG["chunk"]["chunk_size"]
CHUNK_OVERLAP = _RETR_CFG["chunk"]["chunk_overlap"]

# ---- 多知识库配置 ----
KB_CONFIGS_DIR = "kb_configs"   # YAML 知识库配置目录
DEFAULT_KB_ID = "newspaper"  # 启动菜单的默认数据库 ID  # 启动时默认选中的知识库 ID

# ---- 高亮配置（纯展示层功能，不影响索引或检索）----
HIGHLIGHT_TOP_N = _RETR_CFG["highlight"]["top_n"]       # 每个片段高亮几句（按相似度排序取 top-N）
HIGHLIGHT_THRESHOLD = _RETR_CFG["highlight"]["threshold"]  # 相似度低于此值的句子不高亮（避免噪音）
HIGHLIGHT_SENTENCE_SPLIT = r"[。！？\n]+"  # 中文分句正则（按句号/感叹号/问号/换行切分）
HIGHLIGHT_CONFIG_FILE = "highlight_config.json"  # 高亮样式配置文件（颜色/下划线/斜体等）

# 1) 全局配置：LLM 在 main() 里按用户选择初始化；HuggingFace 做 embedding（本地）
# LLM 提供商预设：通过环境变量 LLM_PROVIDER 可跳过菜单
#   deepseek / qwen / zhipu / ollama / openai_like / custom
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").lower()


def _select_llm_provider() -> str:
    """交互式选择 LLM 提供商。若已通过环境变量 LLM_PROVIDER 指定，则直接使用。"""
    providers = [
        ("deepseek", "DeepSeek (deepseek-v4-pro / deepseek-v4-flash)"),
        ("qwen",     "通义千问 (qwen-plus / qwen-turbo)"),
        ("zhipu",    "智谱 GLM (glm-4 / glm-4-flash)"),
        ("ollama",   "Ollama 本地模型 (qwen2.5:7b 等)"),
        ("custom",   "自定义 OpenAI 兼容接口"),
    ]
    if LLM_PROVIDER:
        for key, _ in providers:
            if key == LLM_PROVIDER:
                print(f"[LLM] 使用环境变量指定的提供商: {LLM_PROVIDER}")
                return LLM_PROVIDER
        print(f"[警告] LLM_PROVIDER={LLM_PROVIDER} 不在支持列表，进入交互选择。")

    print("\n请选择 LLM 提供商：")
    for i, (_, desc) in enumerate(providers, 1):
        print(f"  {i}. {desc}")
    while True:
        try:
            choice = input("输入序号 (1-5)，默认 1: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(providers):
            return providers[int(choice) - 1][0]
        print(f"无效输入: {choice}，请输入 1-{len(providers)} 的数字")


def _build_llm(provider: str):
    """根据提供商构建 LLM 实例。所有 API Key 从环境变量读取。"""
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            sys.exit("错误：请设置环境变量 DEEPSEEK_API_KEY")
        return DeepSeek(model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"), api_key=api_key, temperature=0.0, max_tokens=512)

    if provider == "qwen":
        # 阿里云通义千问：OpenAI 兼容接口
        if OpenAILike is None:
            sys.exit("错误：未安装 llama-index-llms-openai-like，请运行 pip install llama-index-llms-openai-like")
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            sys.exit("错误：请设置环境变量 DASHSCOPE_API_KEY（阿里云 DashScope）")
        return OpenAILike(
            model=os.environ.get("QWEN_MODEL", "qwen-plus"),
            api_key=api_key,
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
            max_tokens=512,
        )

    if provider == "zhipu":
        # 智谱 GLM：OpenAI 兼容接口
        if OpenAILike is None:
            sys.exit("错误：未安装 llama-index-llms-openai-like")
        api_key = os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            sys.exit("错误：请设置环境变量 ZHIPU_API_KEY")
        return OpenAILike(
            model=os.environ.get("ZHIPU_MODEL", "glm-4"),
            api_key=api_key,
            api_base="https://open.bigmodel.cn/api/paas/v4",
            temperature=0.0,
            max_tokens=512,
        )

    if provider == "ollama":
        if Ollama is None:
            sys.exit("错误：未安装 llama-index-llms-ollama")
        return Ollama(
            model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0,
            request_timeout=120.0,
        )

    if provider in ("custom", "openai_like"):
        if OpenAILike is None:
            sys.exit("错误：未安装 llama-index-llms-openai-like")
        api_key = os.environ.get("CUSTOM_API_KEY", "EMPTY")
        return OpenAILike(
            model=os.environ.get("CUSTOM_MODEL", "gpt-3.5-turbo"),
            api_key=api_key,
            api_base=os.environ.get("CUSTOM_API_BASE", "http://localhost:8000/v1"),
            temperature=0.0,
            max_tokens=512,
        )

    sys.exit(f"错误：未知 LLM 提供商 {provider}")


# 启动时按用户选择初始化 LLM（在 main() 调用）
# Settings.llm 会在 main() 里设置
# ---- Embedding 模型初始化 ----
# Qwen2-7B 等大模型需要特殊配置：trust_remote_code、query_instruction、FP16、last_token pooling
# bge 系列等小模型用基础配置即可
# 启动时扫描本地可用模型 + 交互选择；也可通过 EMBED_MODEL 环境变量跳过菜单

# 本地模型搜索位置
_LOCAL_MODEL_DIRS = [
    r"C:\code\LlamaIndex\models",            # 项目内本地模型
    os.path.expanduser("~/.cache/huggingface/hub"),  # HF 缓存
]


def _scan_local_embed_models() -> list:
    """扫描本地已下载的 embedding 模型，返回 [(显示名, 路径或HF ID), ...]"""
    found = []
    seen = set()

    # 1) 项目内本地模型目录（完整模型文件，直接用路径加载）
    local_dir = _LOCAL_MODEL_DIRS[0]
    if os.path.isdir(local_dir):
        for name in sorted(os.listdir(local_dir)):
            full = os.path.join(local_dir, name)
            # 必须含 config.json 才算完整模型
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
                if full not in seen:
                    seen.add(full)
                    found.append((f"{name} (本地路径)", full))

    # 2) HF 缓存中的模型（按 models--org--name 结构存）
    hf_cache = _LOCAL_MODEL_DIRS[1]
    if os.path.isdir(hf_cache):
        for sub in sorted(os.listdir(hf_cache)):
            if not sub.startswith("models--"):
                continue
            # models--BAAI--bge-small-zh -> BAAI/bge-small-zh
            parts = sub.split("--")
            if len(parts) >= 3:
                hf_id = "/".join(parts[1:])
                # 必须有 snapshot 子目录才算下载完整
                snap = os.path.join(hf_cache, sub, "snapshots")
                if os.path.isdir(snap) and hf_id not in seen:
                    seen.add(hf_id)
                    found.append((f"{hf_id} (HF缓存)", hf_id))

    return found


def _is_qwen2_embed(model_path_or_id: str) -> bool:
    return "qwen2" in model_path_or_id.lower() or "gte-qwen" in model_path_or_id.lower()


def _select_embed_model() -> str:
    """交互式选择 embedding 模型。若 EMBED_MODEL 环境变量已指定且非空，则直接使用。"""
    # 环境变量预设优先
    if EMBED_MODEL:
        # 如果 EMBED_MODEL 等于默认值，也进入菜单（让用户能切换）
        # 但如果用户明确设置了非默认值，则直接使用
        if os.environ.get("EMBED_MODEL"):
            print(f"[Embedding] 使用环境变量指定的模型: {EMBED_MODEL}")
            return EMBED_MODEL

    local_models = _scan_local_embed_models()
    # 默认选 bge-m3（中文好+速度快），找不到则回退第 1 个
    default_idx = 1
    for i, (desc, path) in enumerate(local_models, 1):
        if "bge-m3" in desc.lower():
            default_idx = i
            break
    print("\n请选择 embedding 模型：")
    print("  0. 手动输入模型名/路径（如 BAAI/bge-m3 或 D:\\path\\to\\model）")
    for i, (desc, path) in enumerate(local_models, 1):
        marker = " (默认)" if i == default_idx else ""
        print(f"  {i}. {desc}{marker}")
    while True:
        try:
            choice = input(f"输入序号 (0-{len(local_models)})，默认 {default_idx}: ").strip() or str(default_idx)
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if choice == "0":
            try:
                manual = input("输入模型名或本地路径: ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)
            if manual:
                return manual
            print("输入为空，请重新选择")
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(local_models):
            return local_models[int(choice) - 1][1]
        print(f"无效输入: {choice}，请输入 0-{len(local_models)} 的数字")


def _build_embed_model(model_path_or_id: str):
    """根据模型路径/ID 构建 HuggingFaceEmbedding 实例，自动适配 Qwen2 和小模型配置"""
    if _is_qwen2_embed(model_path_or_id):
        # Qwen2 系列 embedding 模型（gte-Qwen2-7B-instruct 等）的 query 端必须加 instruction prefix
        # 否则检索质量会显著下降（官方文档明确要求）
        _QWEN2_QUERY_INSTRUCTION = (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        )
        return HuggingFaceEmbedding(
            model_name=model_path_or_id,
            trust_remote_code=True,                # 必须：Qwen2 用了自定义代码
            query_instruction=_QWEN2_QUERY_INSTRUCTION,  # query 端加 instruction
            text_instruction=None,                 # 文档端不加（官方推荐）
            max_length=2048,                        # 16GB 显存限制：8192 会 OOM，2048 足够覆盖大多数 chunk
            embed_batch_size=2,                     # 16GB 显存限制：batch=2 配合 max_length=2048 较稳
            model_kwargs={"torch_dtype": torch.float16},  # 必须：FP16 加载省一半显存
        )
    # bge/MiniLM 等小模型用基础配置
    return HuggingFaceEmbedding(
        model_name=model_path_or_id,
        embed_batch_size=64,   # 批量推理，大幅加速 CPU embedding
        model_kwargs={"torch_dtype": torch.float16},  # 默认 FP16 加载，省一半显存
    )


# 启动时在 main() 里调用 _select_embed_model() + _build_embed_model()
# Settings.embed_model 会在 main() 里设置
Settings.chunk_size = CHUNK_SIZE


# ---- Embedding 模型一致性校验 ----
def _get_embed_dim() -> int:
    """获取当前 embed_model 的输出维度。"""
    return len(Settings.embed_model.get_text_embedding("test"))


def _check_embed_consistency(storage_context: StorageContext) -> bool:
    """
    校验现有索引中的向量维度是否与当前 embed_model 一致。

    返回 True 表示一致，False 表示不一致（调用方应提示用户 rebuild）。
    避免查询时 NumPy 报 inhomogeneous shape 错误。
    """
    try:
        emb_dict = storage_context.vector_store.data.embedding_dict
    except AttributeError:
        return True  # 无法访问，跳过检查
    if not emb_dict:
        return True  # 空索引，无需检查

    # 采样第一个向量的维度作为存储维度
    stored_dim = len(next(iter(emb_dict.values())))
    expected_dim = _get_embed_dim()
    if stored_dim != expected_dim:
        print(
            f"\n[警告] 索引中的向量维度（{stored_dim}）与当前 embedding 模型 "
            f"{EMBED_MODEL} 的维度（{expected_dim}）不一致！\n"
            f"这会导致查询时报错。请执行 rebuild 命令重建索引，\n"
            f"或用环境变量 EMBED_MODEL 指回原模型后重启。\n"
        )
        return False
    return True


# ---- 存储容量辅助函数 ----
def get_storage_size(path: str = STORAGE_DIR) -> int:
    """递归计算目录总字节数。不存在返回 0。"""
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def format_size(num_bytes: int) -> str:
    """字节数转人类可读字符串：B / KB / MB / GB。"""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


# ---- 重点句高亮（纯展示层，不影响索引或检索）----

# ANSI 颜色码映射表
_ANSI_COLOR_CODES = {
    "black": 30, "red": 31, "green": 32, "yellow": 33,
    "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
    "bright_black": 90, "bright_red": 91, "bright_green": 92, "bright_yellow": 93,
    "bright_blue": 94, "bright_magenta": 95, "bright_cyan": 96, "bright_white": 97,
}


def _load_highlight_config() -> dict:
    """
    从 highlight_config.json 加载高亮样式配置。

    配置文件不存在时使用默认值（亮黄+下划线），保证程序可独立运行。
    配置文件格式错误时打印警告并使用默认值，不中断主流程。
    """
    default = {
        "foreground": "bright_yellow",
        "background": None,
        "bold": False,
        "italic": False,
        "underline": True,
        "strikethrough": False,
        "reverse": False,
        "fallback_prefix": ">>>",
        "fallback_suffix": "<<<",
    }

    config_path = Path(HIGHLIGHT_CONFIG_FILE)
    if not config_path.exists():
        return default

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            # 过滤掉 _comment 开头的字段
            raw = json.load(f)
            user_cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
        # 合并：用户配置覆盖默认值
        merged = {**default, **user_cfg}
        return merged
    except (json.JSONDecodeError, OSError) as e:
        print(f"[警告] 高亮配置文件解析失败 ({type(e).__name__}: {e})，使用默认样式")
        return default


def _build_ansi_code(cfg: dict) -> str:
    """
    根据配置构建 ANSI 转义码（不含结尾 reset）。

    例如配置 {foreground: bright_yellow, underline: true}
    返回 "4;93"（用 ; 分隔多个 SGR 参数）
    """
    codes = []
    # 样式码（顺序：bold → italic → underline → strikethrough → reverse）
    if cfg.get("bold"):
        codes.append("1")
    if cfg.get("italic"):
        codes.append("3")
    if cfg.get("underline"):
        codes.append("4")
    if cfg.get("strikethrough"):
        codes.append("9")
    if cfg.get("reverse"):
        codes.append("7")

    # 颜色码
    fg = cfg.get("foreground")
    bg = cfg.get("background")
    if fg and fg in _ANSI_COLOR_CODES:
        codes.append(str(_ANSI_COLOR_CODES[fg]))
    if bg and bg in _ANSI_COLOR_CODES:
        # 背景色 = 前景色码 + 10
        codes.append(str(_ANSI_COLOR_CODES[bg] + 10))

    return ";".join(codes) if codes else "0"


def _highlight_relevant_sentences(text: str, query: str) -> str:
    """
    用本地 embed_model 在片段内找出与查询最相似的句子并高亮。

    流程：
      1. 用正则按中文标点切句
      2. 批量计算句子向量与查询向量的余弦相似度
      3. 选 top-N 句（相似度需 >= HIGHLIGHT_THRESHOLD）标记为重点
      4. 输出格式：
         - tty 环境（独立 PowerShell）：根据 highlight_config.json 配置生成 ANSI 样式
         - 非 tty（IDE 捕获）：用 fallback_prefix/fallback_suffix 标记包裹

    样式配置文件：highlight_config.json（颜色/下划线/斜体/加粗等）
    无 API 调用，仅用已加载的本地 embed_model。
    """
    text = text.strip()
    if not text or not query:
        return text

    # 1) 正则分句，保留分隔符以还原原文
    parts = re.split(f"({HIGHLIGHT_SENTENCE_SPLIT})", text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        # parts[i] 是句子，parts[i+1] 是分隔符
        sent = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if sent.strip():
            sentences.append(sent)
    if parts and len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])

    if len(sentences) <= 1:
        return text  # 只有一句，无需高亮

    # 2) 批量向量化
    try:
        sent_embeddings = Settings.embed_model.get_text_embedding_batch(sentences)
        # 重要：必须用 get_query_embedding 而非 get_text_embedding
        # 对于 Qwen2 系列 query-sensitive 模型，前者会自动加 query_instruction
        # 否则查询向量和文档向量不在同一表示空间，相似度计算结果会偏差很大
        query_embedding = Settings.embed_model.get_query_embedding(query)
    except Exception:
        return text  # embedding 失败则原样返回，不影响主流程

    # 3) 计算余弦相似度
    sent_arr = np.array(sent_embeddings)
    query_arr = np.array(query_embedding)
    # 归一化后点积 = 余弦相似度
    norms = np.linalg.norm(sent_arr, axis=1) * np.linalg.norm(query_arr)
    norms[norms == 0] = 1e-10  # 防除零
    similarities = (sent_arr @ query_arr) / norms

    # 4) 选 top-N 句（相似度 >= 阈值）
    top_n = min(HIGHLIGHT_TOP_N, len(sentences))
    top_indices = set()
    for idx in np.argsort(similarities)[::-1][:top_n]:
        if similarities[idx] >= HIGHLIGHT_THRESHOLD:
            top_indices.add(int(idx))

    if not top_indices:
        return text  # 没有句子达到阈值，原样返回

    # 5) 组装输出
    is_tty = sys.stdout.isatty()
    hl_cfg = _load_highlight_config()
    result = []
    for i, sent in enumerate(sentences):
        if i in top_indices:
            if is_tty:
                # 根据 highlight_config.json 动态构建 ANSI 转义码
                ansi_code = _build_ansi_code(hl_cfg)
                result.append(f"\033[{ansi_code}m{sent}\033[0m")
            else:
                # 非 tty 环境：用配置文件里的标记包裹
                prefix = hl_cfg.get("fallback_prefix", ">>>")
                suffix = hl_cfg.get("fallback_suffix", "<<<")
                result.append(f"{prefix}{sent}{suffix}")
        else:
            result.append(sent)
    return "".join(result)


# ---- 分页输出辅助函数 ----
PAGER_THRESHOLD_LINES = 30  # 超过此行数才触发分页，短的直接 print


def print_paged(text: str, page_lines: int = 40) -> None:
    """
    分页显示长文本。

    策略：
      1. 行数 <= PAGER_THRESHOLD_LINES → 直接 print（短内容不打扰用户）
      2. stdin 是真实 tty（用户可交互） → 调用 pydoc.pager（系统 more.com/less，最成熟）
         若 pydoc.pager 失败，回退到手写按行分页（Enter 继续，q 退出）
      3. stdin 非 tty（IDE 捕获/管道重定向/CI 日志） → 直接全打印，
         因为非交互环境无法接收用户按键，强行分页只会刷屏
    """
    lines = text.split("\n")
    if len(lines) <= PAGER_THRESHOLD_LINES:
        print(text)
        return

    # 非交互环境（stdin 非 tty）：无法接收用户输入，直接全打印
    if not sys.stdin.isatty():
        print(text)
        return

    # 真实终端：优先用标准库 pydoc.pager 调用系统 pager（最成熟）
    try:
        import pydoc
        pydoc.pager(text)
        return
    except Exception:
        pass  # 失败则走手写兜底

    # 手写按行分页兜底
    total = len(lines)
    for i in range(0, total, page_lines):
        chunk = lines[i:i + page_lines]
        print("\n".join(chunk))
        if i + page_lines < total:
            try:
                cmd = input(f"\n-- 更多 ({i + page_lines}/{total} 行) [Enter=继续 q=退出] ")
                if cmd.strip().lower() in {"q", "quit", "退出"}:
                    print("-- 已中断显示 --")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\n-- 已中断显示 --")
                return


# ---- 带超时的输入（用于 ingest 批次间确认）----
def _input_with_timeout(prompt: str, timeout: float) -> str:
    """
    带超时的 input。超时返回空字符串。

    跨平台策略：
      - Windows: 用 msvcrt 逐键读取，可精确中断
      - Unix: 用 select 监听 stdin
      - 都不可用：回退到阻塞 input（无超时）

    注意：非 tty 环境（IDE 捕获/管道）会立即返回空字符串，
    避免在无法接收按键的场景下永久阻塞。
    """
    # 非 tty 环境（IDE 捕获/管道重定向）：无法接收用户按键，直接返回空字符串
    # 让调用方按"超时自动继续"的语义处理
    if not sys.stdin.isatty():
        print(prompt, end="", flush=True)
        print(f"\n[非交互环境] 自动继续...")
        return ""

    # Windows: msvcrt
    if os.name == "nt":
        try:
            import msvcrt
            import time
            print(prompt, end="", flush=True)
            start = time.time()
            buf = []
            while True:
                # 检查键盘是否有输入
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode("utf-8", errors="ignore")
                    if ch in ("\r", "\n"):
                        print()
                        return "".join(buf)
                    elif ch == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt
                    elif ch == "\x08":  # Backspace
                        if buf:
                            buf.pop()
                            print("\b \b", end="", flush=True)
                    elif ch.isprintable():
                        buf.append(ch)
                        print(ch, end="", flush=True)
                # 超时检查
                if time.time() - start >= timeout:
                    print(f"\n[超时] {timeout}秒未响应，自动继续...")
                    return ""
                time.sleep(0.05)
        except ImportError:
            pass  # msvcrt 不可用，回退到下面的 select / 阻塞 input

    # Unix: select
    try:
        import select as _select
        print(prompt, end="", flush=True)
        rlist, _, _ = _select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.readline().rstrip("\r\n")
        print(f"\n[超时] {timeout}秒未响应，自动继续...")
        return ""
    except (ImportError, OSError):
        # 最终回退：阻塞 input（无超时，但极少触发）
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return "q"


# ---- 聚合查询：直遍 docstore（跳过向量检索）----
# 背景：向量检索基于语义相似度返回 top-k 片段，适合"找最相关的几段"。
# 但"列出所有X"类聚合查询需要扫描全库，向量检索天然无法覆盖。
# 此模块检测聚合查询，直接遍历 docstore 用正则提取，精度 100%。

# 聚合查询触发模式
_AGGREGATE_PATTERNS = [
    r"列出所有", r"列出全部", r"所有的", r"全部的",
    r"有哪些", r"有什么", r"列举", r"全部列出",
    r"所有.{0,4}标题", r"所有.{0,4}项目", r"所有.{0,4}文章",
    r"列出.{0,4}文章", r"列出.{0,4}标题", r"列出.{0,4}项目",
]


def _is_aggregate_query(question: str) -> bool:
    """检测是否是聚合查询（需要扫描全库而非语义检索的查询）。"""
    for p in _AGGREGATE_PATTERNS:
        if re.search(p, question):
            return True
    return False


def _extract_keyword_from_query(question: str) -> str:
    """从聚合查询中提取目标关键词（去掉"列出所有"等虚词）。"""
    cleaned = re.sub(
        r"列出所有|列出全部|所有的|全部的|有哪些|有什么|列举|全部列出|"
        r"请|吗|呢|啊|吧|\?|？|列出|所有|全部",
        "", question
    )
    return cleaned.strip()


def _handle_aggregate_query(question: str, index) -> bool:
    """
    处理聚合查询：直接遍历 docstore 提取，不走向量检索。

    返回 True 表示已处理（调用方应 continue），False 表示无法处理（走正常流程）。
    遍历全部节点，用正则/关键词匹配提取，精度 100%。
    """
    docstore = index.storage_context.docstore
    docs = docstore.docs
    total = len(docs)

    results = []
    seen = set()

    # 策略1：文章标题（### 或 ## 开头的 markdown 标题）
    if re.search(r"标题|文章名|文章列表|文章目录", question):
        print(f"\n[聚合查询] 提取模式：文章标题（### 开头）| 扫描 {total} 个节点 ...")
        for _, node in tqdm(docs.items(), desc="扫描", total=total, ncols=70, leave=False):
            text = node.get_content()
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("### ") and line not in seen:
                    seen.add(line)
                    results.append(line)

    # 策略2：项目名称（包含"项目"关键词的行）
    elif re.search(r"项目", question):
        print(f"\n[聚合查询] 提取模式：包含「项目」的行 | 扫描 {total} 个节点 ...")
        for _, node in tqdm(docs.items(), desc="扫描", total=total, ncols=70, leave=False):
            text = node.get_content()
            for line in text.split("\n"):
                line = line.strip()
                # 过滤太长的行（正文段落）和太短的行（无意义）
                if "项目" in line and 4 < len(line) < 200 and line not in seen:
                    seen.add(line)
                    results.append(line)

    # 策略3：通用关键词提取
    else:
        keyword = _extract_keyword_from_query(question)
        if not keyword:
            print(f"\n[聚合查询] 无法识别提取目标，回退到正常查询")
            return False
        print(f"\n[聚合查询] 提取模式：包含「{keyword}」的行 | 扫描 {total} 个节点 ...")
        for _, node in tqdm(docs.items(), desc="扫描", total=total, ncols=70, leave=False):
            text = node.get_content()
            for line in text.split("\n"):
                line = line.strip()
                if keyword in line and 4 < len(line) < 200 and line not in seen:
                    seen.add(line)
                    results.append(line)

    # 输出结果
    if results:
        parts = [
            "",
            "=" * 60,
            f"聚合查询结果：共找到 {len(results)} 条（直遍 docstore，非向量检索）",
            "=" * 60,
        ]
        for i, item in enumerate(results, 1):
            parts.append(f"  {i:>3}. {item}")
        parts.append("=" * 60)
        parts.append(f"提示：扫描了全部 {total} 个节点，覆盖率 100%（向量检索 top_k=2 仅覆盖 0.004%）")
        print_paged("\n".join(parts))
    else:
        print(f"\n[聚合查询] 未找到匹配内容（扫描了 {total} 个节点）")

    sys.stdout.flush()
    return True


# ---- 检索模式管理 ----
# 三种检索模式并存，可随时切换：
#   vector    向量检索（语义匹配，LLM 生成回答，top-k 返回）
#   aggregate 聚合直遍（正则匹配，100% 覆盖，适合"列出所有X"类查询）
#   fulltext  全文搜索（倒排索引 + BM25 排序，关键词精确匹配）

RETRIEVAL_MODES = {
    "vector":    "向量检索（语义匹配，LLM 生成回答）",
    "aggregate": "聚合直遍（正则匹配，100%覆盖，适合列举类查询）",
    "fulltext":  "全文搜索（倒排索引 + BM25 排序，关键词精确匹配）",
}


class FullTextSearcher:
    """
    基于倒排索引 + BM25 的全文搜索引擎。

    特点：
      - 中文 bigram 分词（无需 jieba 等外部依赖）
      - 英文按空格/标点分词
      - BM25 排序（考虑词频、文档长度、IDF）
      - 懒加载：首次 search 时才构建倒排索引

    适用场景：关键词精确匹配，比向量检索快且可解释，比聚合直遍更有排序。
    """

    def __init__(self, index):
        self.index = index
        self.inverted_index = {}    # {term: {doc_id: tf}}
        self.doc_lengths = {}       # {doc_id: word_count}
        self.avg_doc_length = 0
        self.total_docs = 0
        self.doc_ids = []           # 有序 doc_id 列表
        self._built = False

    def _tokenize(self, text: str) -> list:
        """
        分词：中文 bigram + 英文单词。

        示例：
          "北京地铁项目" → ["北京", "京地", "地铁", "铁项", "项目", "project"]
          "AI project"   → ["ai", "project"]
        """
        tokens = []
        # 提取连续的中文段落和英文单词
        segments = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())
        for seg in segments:
            if len(seg) >= 2:
                if seg[0] >= '\u4e00':  # 中文段落：bigram
                    for i in range(len(seg) - 1):
                        tokens.append(seg[i:i + 2])
                else:  # 英文单词：整词
                    tokens.append(seg)
        return tokens

    def build(self, show_progress: bool = True):
        """构建倒排索引。遍历 docstore 中所有节点。"""
        docstore = self.index.storage_context.docstore
        docs = docstore.docs
        self.total_docs = len(docs)
        self.doc_ids = list(docs.keys())

        iterator = docs.items()
        if show_progress:
            iterator = tqdm(iterator, desc="构建倒排索引", total=self.total_docs, ncols=70, leave=False)

        for doc_id, node in iterator:
            text = node.get_content()
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)

            # 统计词频
            tf_map = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            # 写入倒排索引
            for token, freq in tf_map.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                self.inverted_index[token][doc_id] = freq

        # 计算平均文档长度
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        self._built = True

    def search(self, query: str, top_k: int = 20) -> list:
        """
        BM25 搜索，返回 [(doc_id, score, node), ...] 按 score 降序。

        BM25 公式：
          score(D, Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))
          IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        """
        if not self._built:
            self.build()

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # BM25 参数
        k1 = _RETR_CFG["fulltext"]["bm25_k1"]  # 词频饱和参数
        b = _RETR_CFG["fulltext"]["bm25_b"]    # 文档长度归一化参数

        # 计算每个文档的 BM25 分数
        scores = {}  # {doc_id: score}
        docstore = self.index.storage_context.docstore

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            postings = self.inverted_index[token]  # {doc_id: tf}
            n_qi = len(postings)  # 包含该词的文档数
            idf = math.log((self.total_docs - n_qi + 0.5) / (n_qi + 0.5) + 1)

            for doc_id, tf in postings.items():
                dl = self.doc_lengths.get(doc_id, 0)
                denom = tf + k1 * (1 - b + b * dl / self.avg_doc_length) if self.avg_doc_length > 0 else tf + k1
                score = idf * (tf * (k1 + 1)) / denom
                scores[doc_id] = scores.get(doc_id, 0) + score

        # 排序取 top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 关联 node 对象
        results = []
        for doc_id, score in ranked:
            node = docstore.docs.get(doc_id)
            if node:
                results.append((doc_id, score, node))

        return results

    @property
    def status(self) -> str:
        """返回索引构建状态字符串。"""
        if not self._built:
            return "未构建"
        return f"已构建（{self.total_docs} 文档, {len(self.inverted_index)} 词项, 平均长度 {self.avg_doc_length:.0f}）"


# 全局全文搜索器实例（懒加载）
_fulltext_searcher = None


def _get_fulltext_searcher(index) -> FullTextSearcher:
    """获取或构建全局全文搜索器实例。"""
    global _fulltext_searcher
    if _fulltext_searcher is None or _fulltext_searcher.index is not index:
        _fulltext_searcher = FullTextSearcher(index)
    return _fulltext_searcher


def _handle_fulltext_query(question: str, index, top_k: int = _RETR_CFG["fulltext"]["top_k"]) -> bool:
    """
    处理全文搜索查询：BM25 排序，返回匹配片段。

    返回 True（总是已处理）。
    与向量检索的区别：不调 LLM，直接返回匹配片段，快且可解释。
    与聚合直遍的区别：有 BM25 排序，按相关性返回 top-k，不是全量列出。
    """
    searcher = _get_fulltext_searcher(index)

    if not searcher._built:
        print(f"\n[全文搜索] 首次使用，正在构建倒排索引 ...")
        import time
        t0 = time.time()
        searcher.build()
        print(f"    -> 构建完成，耗时 {time.time()-t0:.1f}s | {searcher.status}")

    print(f"\n[全文搜索] BM25 搜索 top_k={top_k} ...")
    import time
    t0 = time.time()
    results = searcher.search(question, top_k=top_k)
    elapsed = time.time() - t0

    if not results:
        print(f"    -> 未找到匹配文档（耗时 {elapsed:.2f}s）")
        return True

    # 统计查询词在结果中的命中情况
    query_tokens = searcher._tokenize(question)

    parts = [
        "",
        "=" * 60,
        f"全文搜索结果：共 {len(results)} 条（BM25 排序，top_k={top_k}）",
        f"查询词分词: {' / '.join(query_tokens)} | 搜索耗时 {elapsed:.3f}s",
        "=" * 60,
    ]

    for i, (doc_id, score, node) in enumerate(results, 1):
        meta = node.metadata or {}
        file_path = meta.get("file_path", "未知来源")
        content = node.get_content()
        # 高亮匹配的关键词
        highlighted = _highlight_keywords(content, query_tokens)
        preview = highlighted[:300] + ("..." if len(content) > 300 else "")
        parts.append(f"\n--- 结果 {i} | BM25 score={score:.4f} | 来源: {file_path} ---")
        parts.append(preview)

    parts.append("\n" + "=" * 60)
    parts.append(f"提示：全文搜索按 BM25 相关性排序，不调 LLM。用 mode 命令切换到向量检索获取 LLM 回答。")
    print_paged("\n".join(parts))

    sys.stdout.flush()
    return True


def _highlight_keywords(text: str, keywords: list) -> str:
    """在文本中高亮命中的关键词（用 ANSI 亮黄色 + 下划线）。"""
    if not sys.stdout.isatty():
        # 非 tty：用 >>> <<< 标记
        result = text
        for kw in keywords:
            if kw and len(kw) >= 2:
                result = result.replace(kw, f">>>{kw}<<<")
        return result

    # tty：ANSI 高亮
    HL = "\033[93;4m"  # 亮黄色 + 下划线
    RESET = "\033[0m"
    result = text
    for kw in keywords:
        if kw and len(kw) >= 2:
            result = result.replace(kw, f"{HL}{kw}{RESET}")
    return result


def _select_retrieval_mode() -> str:
    """交互式选择检索模式。"""
    mode_keys = list(RETRIEVAL_MODES.keys())
    print("\n请选择检索模式：")
    for i, key in enumerate(mode_keys, 1):
        marker = " (默认)" if key == "vector" else ""
        print(f"  {i}. {RETRIEVAL_MODES[key]}{marker}")
    print("  提示：运行中可用 'mode' 命令随时切换")
    while True:
        try:
            choice = input(f"输入序号 (1-{len(mode_keys)})，默认 1: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(mode_keys):
            return mode_keys[int(choice) - 1]
        print(f"无效输入: {choice}，请输入 1-{len(mode_keys)} 的数字")


# ---- SimHash 近似转载去重 ----
# 背景：DocstoreStrategy.DUPLICATES_ONLY 用的 Document.hash 把 metadata（含 file_path）
# 一并算进哈希——同一篇文章只要来自不同文件哈希就不同，报纸之间的转载完全拦不住。
# 这里在摄入前对正文做内容指纹：NFKC 归一化 + 去标点空白 → 字符 bigram → 64 位 SimHash，
# 汉明距离 <= 阈值（默认 3，retrieval_config.yaml 的 dedup 区可调）即判定为转载并跳过。
# 指纹持久化到 <kb_storage>/simhash_fp.json（含被拦截转载的来源记录）；
# 老库首次 ingest 时自动从 docstore 已有 chunk 重建指纹（一次性）。
_SIMHASH_FILE = "simhash_fp.json"
_SIMHASH_SEG_MASK = (1 << 16) - 1
_SIMHASH_BIT_SHIFTS = np.arange(64, dtype=np.uint64)


def _normalize_for_simhash(text: str) -> str:
    """归一化正文：NFKC（全/半角、兼容字符统一）后去除所有非文字字符（标点/空白）。"""
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text))


def _simhash64(norm_text: str, min_chars: int = 8):
    """对归一化文本计算 64 位 SimHash（字符 bigram + blake2b，numpy 向量化投票）。
    文本太短（bigram 数 < min_chars）时指纹不可靠，返回 None（调用方退化为精确哈希匹配）。"""
    n = len(norm_text) - 1
    if n < min_chars:
        return None
    hashes = np.fromiter(
        (int.from_bytes(hashlib.blake2b(norm_text[i:i + 2].encode("utf-8"), digest_size=8).digest(), "little")
         for i in range(n)),
        dtype=np.uint64, count=n,
    )
    bits = (hashes[:, None] >> _SIMHASH_BIT_SHIFTS) & np.uint64(1)
    votes = bits.sum(axis=0, dtype=np.int64) * 2 - n
    fp = 0
    for b in np.flatnonzero(votes > 0):
        fp |= 1 << int(b)
    return fp


class _SimHashStore:
    """SimHash 指纹库：内存索引 + JSON 持久化（每个知识库一个文件）。

    查找用 4×16bit 分段索引：汉明距离 <=3 的两个 64 位指纹至少有一个 16 位段完全相同
    （鸽笼原理），因此只需查 4 个段表取候选再精确算距离，开销与库规模近似无关。
    每个指纹记录首发文件路径与被拦截的转载文件列表（dups），供统计和将来"来源合并"用。
    文本过短的文档只按归一化精确哈希去重（short_docs）。
    """

    def __init__(self, path, threshold: int = 3, min_chars: int = 8):
        self.path = Path(path)
        self.threshold = threshold
        self.min_chars = min_chars
        self.entries = {}       # fp(int)         -> {"file_path", "exact", "dups": []}
        self.short_docs = {}    # exact_hash(str) -> {"file_path", "dups": []}
        self._exact_index = {}  # exact_hash -> fp
        self._seg_tables = [{} for _ in range(4)]  # 段值 -> set(fp)

    # ---- 内部 ----
    @staticmethod
    def _segments(fp: int) -> list:
        return [(fp >> (16 * i)) & _SIMHASH_SEG_MASK for i in range(4)]

    def _index_fp(self, fp: int) -> None:
        for i, seg in enumerate(self._segments(fp)):
            self._seg_tables[i].setdefault(seg, set()).add(fp)

    @staticmethod
    def _exact_hash(norm_text: str) -> str:
        return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

    # ---- 统计 ----
    @property
    def count(self) -> int:
        return len(self.entries) + len(self.short_docs)

    @property
    def total_dups(self) -> int:
        return sum(len(e["dups"]) for e in self.entries.values()) + \
            sum(len(e["dups"]) for e in self.short_docs.values())

    # ---- 持久化 ----
    def load(self) -> None:
        """从 JSON 加载并重建段索引。文件缺失=空库；损坏时警告后从空库开始（不阻断 ingest）。"""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for fp_hex, e in (data.get("entries") or {}).items():
                fp = int(fp_hex, 16)
                e.setdefault("dups", [])
                self.entries[fp] = e
                self._exact_index[e.get("exact", "")] = fp
                self._index_fp(fp)
            for k, e in (data.get("short_docs") or {}).items():
                e.setdefault("dups", [])
                self.short_docs[k] = e
        except Exception as ex:
            print(f"[警告] SimHash 指纹库损坏（{type(ex).__name__}: {ex}），将从空库重新开始")
            self.entries.clear()
            self.short_docs.clear()
            self._exact_index.clear()
            self._seg_tables = [{} for _ in range(4)]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "threshold": self.threshold,
            "min_chars": self.min_chars,
            "entries": {format(fp, "016x"): e for fp, e in self.entries.items()},
            "short_docs": self.short_docs,
        }
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ---- 查重 / 入库 ----
    def check_text(self, text: str):
        """查重。返回 (matched_key, dist, kept_path)；无重复返回 (None, -1, "")。
        matched_key 为命中的指纹（int）或短文档精确哈希（str），供 record_dup 使用。"""
        norm = _normalize_for_simhash(text)
        if not norm:
            return None, -1, ""
        exact = self._exact_hash(norm)
        fp = self._exact_index.get(exact)
        if fp is not None:
            return fp, 0, self.entries[fp]["file_path"]
        if exact in self.short_docs:
            return exact, 0, self.short_docs[exact]["file_path"]
        fp = _simhash64(norm, self.min_chars)
        if fp is None:
            return None, -1, ""
        candidates = set()
        for i, seg in enumerate(self._segments(fp)):
            candidates |= self._seg_tables[i].get(seg, set())
        best_fp, best_dist = None, -1
        for cand in candidates:
            d = bin(fp ^ cand).count("1")
            if d <= self.threshold and (best_dist < 0 or d < best_dist):
                best_fp, best_dist = cand, d
        if best_fp is not None:
            return best_fp, best_dist, self.entries[best_fp]["file_path"]
        return None, -1, ""

    def add_text(self, text: str, file_path: str) -> None:
        """把已确认入库的文档写入指纹库。"""
        norm = _normalize_for_simhash(text)
        if not norm:
            return
        exact = self._exact_hash(norm)
        fp = _simhash64(norm, self.min_chars)
        if fp is None:
            self.short_docs.setdefault(exact, {"file_path": file_path, "dups": []})
            return
        if fp in self.entries:
            return
        self.entries[fp] = {"file_path": file_path, "exact": exact, "dups": []}
        self._exact_index[exact] = fp
        self._index_fp(fp)

    def record_dup(self, matched_key, dup_path: str) -> None:
        """记录一次被拦截的转载（挂在被保留文章的 dups 列表上，上限 100 条防膨胀）。"""
        entry = self.entries.get(matched_key)
        if entry is None:
            entry = self.short_docs.get(matched_key)
        if entry is None or not dup_path:
            return
        if dup_path != entry["file_path"] and dup_path not in entry["dups"] and len(entry["dups"]) < 100:
            entry["dups"].append(dup_path)


def _bootstrap_simhash_from_docstore(docstore, store: _SimHashStore) -> int:
    """老库升级：从 docstore 已有 chunk 重建文档级指纹（按 file_path 分组、沿 NEXT 链拼接）。
    仅在指纹库为空且索引已有数据时执行一次，返回重建的文档数。"""
    from llama_index.core.schema import NodeRelationship
    by_file = {}
    for node in docstore.docs.values():
        fp = (node.metadata or {}).get("file_path")
        if fp:
            by_file.setdefault(fp, []).append(node)
    for file_path, nodes in tqdm(by_file.items(), desc="重建转载指纹", unit="篇", ncols=80):
        node_map = {n.node_id: n for n in nodes}

        def _rel(n, rel):
            info = (getattr(n, "relationships", {}) or {}).get(rel)
            return getattr(info, "node_id", None)

        heads = [n for n in nodes if _rel(n, NodeRelationship.PREVIOUS) not in node_map]
        next_map = {n.node_id: _rel(n, NodeRelationship.NEXT) for n in nodes}
        ordered, seen = [], set()
        for head in heads:
            cur = head.node_id
            while cur in node_map and cur not in seen:
                seen.add(cur)
                ordered.append(node_map[cur])
                cur = next_map.get(cur)
        # NEXT 链缺失时退化为无序拼接，保证不丢内容（SimHash 对局部顺序不敏感）
        ordered.extend(n for n in nodes if n.node_id not in seen)
        store.add_text("".join(n.get_text() for n in ordered), file_path)
    return len(by_file)


def _get_simhash_store(storage_dir: str, docstore=None):
    """按配置加载当前知识库的 SimHash 指纹库；老库首次使用自动从 docstore 重建指纹。
    dedup.simhash_enabled=false 时返回 None。CLI 与 Web 共用。"""
    dedup_cfg = _RETR_CFG.get("dedup", {})
    if not dedup_cfg.get("simhash_enabled", True):
        return None
    store = _SimHashStore(
        Path(storage_dir) / _SIMHASH_FILE,
        threshold=dedup_cfg.get("simhash_threshold", 3),
        min_chars=dedup_cfg.get("simhash_min_chars", 8),
    )
    store.load()
    if store.count == 0 and docstore is not None and len(docstore.docs) > 0:
        rebuilt = _bootstrap_simhash_from_docstore(docstore, store)
        if rebuilt:
            store.save()
            print(f"    -> 已从现有索引重建 {rebuilt} 篇文章的转载指纹（一次性）")
    return store


def _filter_near_duplicates(documents: list, store: _SimHashStore):
    """批量转载过滤（Web 摄入用）：返回 (保留文档, 被跳过的 [(doc, kept_path, dist)])。
    保留文档的指纹立即写入 store，同批之内的转载也能拦截。"""
    kept, skipped = [], []
    for doc in documents:
        text = doc.get_text()
        key, dist, kept_path = store.check_text(text)
        if key is not None:
            store.record_dup(key, (doc.metadata or {}).get("file_path", ""))
            skipped.append((doc, kept_path, dist))
        else:
            store.add_text(text, (doc.metadata or {}).get("file_path", ""))
            kept.append(doc)
    return kept, skipped


# ---- IngestionPipeline 工厂函数 ----
def make_pipeline(storage_context: StorageContext) -> IngestionPipeline:
    """
    创建带哈希去重的 IngestionPipeline，复用已有 docstore。

    注意：不传 vector_store 给 pipeline。pipeline 只负责去重 + 切块 + embedding，
    返回已嵌入的节点。由调用方（VectorStoreIndex 或 index.insert_nodes）负责
    写入 vector_store 并更新 index_struct。这是 LlamaIndex 官方推荐模式，
    避免 SimpleVectorStore.stores_text=False 导致 from_vector_store 报错。
    """
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
            Settings.embed_model,
        ],
        docstore=storage_context.docstore,
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )


def _read_documents_with_progress(reader: SimpleDirectoryReader, data_dir: str) -> list:
    """
    带实时进度显示的文档读取（ANSI 控制码 Live 显示）：
      - 顶部：固定总进度条（百分比 + 计数 + 耗时）
      - 底部：最近 5 个文件名同步刷新

    Windows 10+ / PowerShell 5+ 兼容。
    """
    total = len(reader.input_files)
    documents: list = []
    recent: list = []
    max_recent = 5
    import time
    t0 = time.time()

    is_tty = sys.stdout.isatty()

    # Windows 10+ 启用 ANSI 虚拟终端处理（仅终端模式）
    if is_tty and os.name == "nt":
        os.system("")

    print(f"    -> 发现 {total} 个 .md 文件，开始逐个读取...")

    # 预留显示区域：1 行进度条 + max_recent 行文件列表
    display_lines = max_recent + 1
    if is_tty:
        sys.stdout.write("\n" * display_lines)

    for i, docs in enumerate(reader.iter_data(), 1):
        documents.extend(docs)

        # 获取当前文件相对路径
        short = ""
        if docs:
            fname = docs[0].metadata.get("file_path", "") or getattr(docs[0], "id_", "")
            try:
                rel = Path(fname).relative_to(data_dir)
                short = str(rel)
            except Exception:
                short = Path(fname).name if fname else ""
            if len(short) > 65:
                short = "..." + short[-62:]
        recent.append(short)
        if len(recent) > max_recent:
            recent.pop(0)

        # 进度计算
        pct = i / total * 100 if total > 0 else 100
        elapsed = time.time() - t0
        speed = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / speed if speed > 0 else 0

        if is_tty:
            # 终端模式：ANSI Live 显示
            sys.stdout.write(f"\033[{display_lines}A")
            for _ in range(display_lines):
                sys.stdout.write("\033[K\n")
            sys.stdout.write(f"\033[{display_lines}A")

            bar_len = 30
            filled = int(bar_len * i / total) if total > 0 else bar_len
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(
                f"  读取进度: [{bar}] {i}/{total} ({pct:.1f}%) | "
                f"{speed:.1f}file/s | ETA {int(eta)}s\n"
            )
            for j, f in enumerate(recent):
                marker = "▶" if j == len(recent) - 1 else " "
                sys.stdout.write(f"  {marker} {f}\n")
            for _ in range(max_recent - len(recent)):
                sys.stdout.write("\n")
        else:
            # 非终端模式（重定向/管道）：每 50 个文件或最后一个打印一行
            if i % 50 == 0 or i == total:
                print(f"  读取进度: {i}/{total} ({pct:.1f}%) | {speed:.1f}file/s | ETA {int(eta)}s | 当前: {short}")

        sys.stdout.flush()

    # 最终输出完成信息
    if is_tty:
        sys.stdout.write(f"\033[{display_lines}A")
        for _ in range(display_lines):
            sys.stdout.write("\033[K\n")
        sys.stdout.write(f"\033[{display_lines}A")
    elapsed = time.time() - t0
    print(f"  [OK] 读取完成: {total} 个文件，{len(documents)} 个文档 | 耗时 {elapsed:.1f}s")
    sys.stdout.flush()  # 确保终端输出完整，避免 input() 卡顿

    return documents


# ---- 多知识库管理 ----
def _load_kb_configs() -> dict:
    """扫描 kb_configs/ 目录，加载所有 .yaml 配置（不含 .example）。
    返回 {kb_id: {name, description, storage_dir, file_exts}}。
    """
    configs = {}
    cfg_dir = Path(KB_CONFIGS_DIR)
    if not cfg_dir.exists():
        return configs
    try:
        import yaml
    except ImportError:
        print("[警告] 未安装 PyYAML，多知识库功能不可用")
        return configs
    for yml in sorted(cfg_dir.glob("*.yaml")):
        kb_id = yml.stem
        try:
            with open(yml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg["storage_dir"] = cfg.get("storage_dir") or f"./storage/{kb_id}"
            cfg["file_exts"] = cfg.get("file_exts") or [".md"]
            configs[kb_id] = cfg
        except Exception as e:
            print(f"[警告] 加载知识库配置失败 {yml.name}: {e}")
    return configs


def _select_knowledge_base() -> tuple:
    """交互式选择知识库。返回 (kb_id, cfg_dict)。
    若只有一个知识库则直接使用；无配置则回退到默认 DATA_DIR/STORAGE_DIR。
    一直回车会使用 DEFAULT_KB_ID（找不到则第 1 个）。
    """
    configs = _load_kb_configs()

    # 无配置：回退到旧的单库模式
    if not configs:
        print(f"[知识库] 未找到 {KB_CONFIGS_DIR}/*.yaml，使用默认 DATA_DIR")
        return "_default", {
            "name": "默认",
            "description": f"{DATA_DIR}",
            "data_dir": DATA_DIR,
            "storage_dir": STORAGE_DIR,
            "file_exts": [".md"],
        }

    # 只有一个知识库：直接用
    if len(configs) == 1:
        kb_id, cfg = next(iter(configs.items()))
        print(f"[知识库] 仅有一个配置，自动选择: {cfg.get('name', kb_id)}")
        return kb_id, cfg

    # 多个知识库：交互选择
    kb_ids = list(configs.keys())
    default_idx = 1
    for i, kid in enumerate(kb_ids, 1):
        if kid == DEFAULT_KB_ID:
            default_idx = i
            break

    print("\n请选择知识库：")
    for i, kid in enumerate(kb_ids, 1):
        cfg = configs[kid]
        marker = " (默认)" if i == default_idx else ""
        name = cfg.get("name", kid)
        desc = cfg.get("description", "")
        print(f"  {i}. {name}{marker}  [{kid}]")
        if desc:
            print(f"     {desc}")
    while True:
        try:
            choice = input(f"输入序号 (1-{len(kb_ids)})，默认 {default_idx}: ").strip() or str(default_idx)
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(kb_ids):
            kb_id = kb_ids[int(choice) - 1]
            return kb_id, configs[kb_id]
        print(f"无效输入: {choice}，请输入 1-{len(kb_ids)} 的数字")


def _load_or_build_kb(kb_id: str, cfg: dict) -> VectorStoreIndex:
    """加载已有索引或创建空索引（等待用户 ingest）。
    每个类别库独立 storage 目录。无 data_dir 绑定，ingest 时由用户指定文件。
    """
    kb_storage = cfg["storage_dir"]

    if Path(kb_storage).exists() and (Path(kb_storage) / "docstore.json").exists():
        print(f"[加载] 知识库 [{cfg.get('name', kb_id)}] 从 {kb_storage} 读取索引 ...")
        storage_size = get_storage_size(kb_storage)
        print(f"    存储容量: {format_size(storage_size)}")

        # LlamaIndex 的 StorageContext.from_defaults / load_index_from_storage 是原子操作，
        # 无法在内部插桩。用基于存储大小的预估时间 + 后台线程模拟进度条，
        # 最多推进到 95%，实际加载完成后跳到 100%。"相对准确"即可。
        import time
        import threading

        # 预估速率 20 MB/s（JSON 反序列化 + 对象构建综合速率，偏保守）
        estimated_time = max(storage_size / (20 * 1024 * 1024), 1.5)
        is_tty = sys.stdout.isatty()
        stop_event = threading.Event()

        def _progress():
            t0 = time.time()
            while not stop_event.is_set():
                elapsed = time.time() - t0
                if elapsed <= estimated_time:
                    pct = elapsed / estimated_time * 90
                else:
                    # 超过预估时间，缓慢推进到 95%（避免卡在 90%）
                    pct = min(90 + (elapsed - estimated_time) * 2, 95)
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stdout.write(f"\r    加载进度: [{bar}] {pct:.0f}% | {elapsed:.0f}s")
                sys.stdout.flush()
                time.sleep(0.2)

        if is_tty:
            pt = threading.Thread(target=_progress, daemon=True)
            pt.start()
        else:
            print(f"    加载中（预计 {estimated_time:.0f}s）...")

        t0_load = time.time()
        load_error = None
        try:
            storage_context = StorageContext.from_defaults(persist_dir=kb_storage)
            _check_embed_consistency(storage_context)
            index = load_index_from_storage(storage_context, index_id=INDEX_ID)
        except Exception as e:
            load_error = e
        finally:
            if is_tty:
                stop_event.set()
                pt.join()

        if load_error is None:
            elapsed = time.time() - t0_load
            if is_tty:
                sys.stdout.write(f"\r    加载进度: [{'█' * 30}] 100% | {elapsed:.1f}s\n")
                sys.stdout.flush()
            else:
                print(f"    [OK] 加载完成，耗时 {elapsed:.1f}s")
            return index

        # 索引文件损坏（如上次写盘被中断，留下 0 字节/截断的 JSON）：
        # 与其崩溃，不如提示后重置为空库，用户重新 ingest 即可恢复。
        print(f"\n[警告] 知识库 [{cfg.get('name', kb_id)}] 索引文件损坏（{type(load_error).__name__}: {load_error}）")
        print(f"    已将 {kb_storage} 重置为空库，请重新 ingest 数据。")
        import shutil
        shutil.rmtree(kb_storage, ignore_errors=True)

    # 无索引：创建空索引，等待用户手动 ingest
    print(f"[空库] 知识库 [{cfg.get('name', kb_id)}] 暂无数据，已创建空索引")
    print(f"    存储到: {kb_storage}")
    print(f"    使用 ingest <路径> 命令添加文档")
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes=[], storage_context=storage_context)
    index.set_index_id(INDEX_ID)
    Path(kb_storage).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(kb_storage)
    sys.stdout.flush()
    return index


def load_or_build_index() -> VectorStoreIndex:
    """若 ./storage 已存在则直接加载；否则读数据源文档，经 IngestionPipeline 去重+向量化后构建并持久化。"""
    if Path(STORAGE_DIR).exists() and (Path(STORAGE_DIR) / "docstore.json").exists():
        print(f"[加载] 从 {STORAGE_DIR} 读取已持久化的索引 ...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        _check_embed_consistency(storage_context)  # 维度不一致则直接退出
        return load_index_from_storage(storage_context, index_id=INDEX_ID)

    print(f"[构建] 读取数据源: {DATA_DIR}")
    print("      只读取 .md 文件（自动跳过 .obsidian 配置目录等）")
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".md"],       # 只读 markdown 笔记
        recursive=True,              # 递归子目录
        exclude=[".obsidian"],       # 排除 Obsidian 配置目录
        filename_as_id=True,         # 用文件名作为 doc_id，便于调试和 list
    )
    documents = _read_documents_with_progress(reader, DATA_DIR)

    # 限制文档数量（调试用）：MAX_DOCS=10 只处理前 10 个文档
    max_docs = int(os.environ.get("MAX_DOCS", "0"))
    if max_docs > 0 and len(documents) > max_docs:
        print(f"    [调试] MAX_DOCS={max_docs}：只处理前 {max_docs} 个文档（共 {len(documents)} 个）")
        documents = documents[:max_docs]

    storage_context = StorageContext.from_defaults()
    pipeline = make_pipeline(storage_context)

    print(f"[构建] 运行 IngestionPipeline（哈希去重 + 切块 + 向量化）...")
    nodes = pipeline.run(documents=documents, show_progress=True)
    hashes_after = len(storage_context.docstore.get_all_document_hashes())

    # nodes 已由 pipeline 完成 embedding，VectorStoreIndex 会跳过已有 embedding 不重复计算
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )
    index.set_index_id(INDEX_ID)
    index.storage_context.persist(STORAGE_DIR)
    total_size = get_storage_size(STORAGE_DIR)
    print(f"    -> 索引已持久化到 {STORAGE_DIR}，下次启动将直接加载")
    print(f"    -> 输入 {len(documents)} 个文档，产出 {len(nodes)} 个节点，docstore 共跟踪 {hashes_after} 个唯一哈希")
    print(f"    -> 存储容量: {format_size(total_size)}")
    return index


def _create_new_kb(configs: dict) -> tuple:
    """交互式新建一个数据库，保存为 yaml 供下次使用。
    返回 (kb_id, cfg)。新建的库自动加入 configs。
    data_dir 可选（留空则纯靠 ingest 添加文档）。
    """
    print("\n" + "-" * 60)
    print(" 新建数据库")
    print("-" * 60)
    print("提示：不同数据库彼此独立，各有独立的存储目录。")
    print("      新建后会保存为 kb_configs/<kb_id>.yaml，下次启动可直接选择。")
    print("      数据源可选——留空则创建空库，后续用 ingest 添加文档。")
    print("-" * 60)

    # 1) 输入 kb_id（必须，不能重复，不能含特殊字符）
    while True:
        try:
            kb_id = input("数据库 ID（英文/数字/下划线，如 newspaper）: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if not kb_id:
            print("  ID 不能为空，请重新输入")
            continue
        if not kb_id.replace("_", "").isalnum() or not kb_id[0].isalpha():
            print("  ID 只能包含字母/数字/下划线，且以字母开头")
            continue
        if kb_id in configs:
            print(f"  ID 已存在: {kb_id}，请换一个")
            continue
        break

    # 2) 输入显示名（默认 = kb_id）
    try:
        name = input(f"显示名（默认 {kb_id}）: ").strip() or kb_id
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)

    # 3) 输入描述（可选）
    try:
        description = input(f"描述（可选，回车跳过）: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)

    # 4) 文件扩展名（默认 .md）
    try:
        exts_input = input("读取的文件扩展名（逗号分隔，默认 .md）: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)
    if exts_input:
        file_exts = [e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in exts_input.split(",")]
    else:
        file_exts = [".md"]

    # 5) 自动生成 storage_dir
    storage_dir = f"./storage/{kb_id}"

    # 6) 组装 cfg（不绑定 data_dir）
    cfg = {
        "name": name,
        "description": description or "",
        "storage_dir": storage_dir,
        "file_exts": file_exts,
    }

    # 7) 保存为 yaml
    yaml_path = Path(KB_CONFIGS_DIR) / f"{kb_id}.yaml"
    try:
        import yaml
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_data = {
            "name": name,
            "description": cfg["description"],
            "file_exts": file_exts,
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        print(f"\n[已保存] {yaml_path}")
    except Exception as e:
        print(f"[警告] 保存 yaml 失败: {e}（本次会话仍可用，但下次启动需重新输入）")

    print(f"[新建数据库] {name} [{kb_id}]")
    print(f"  存储到: {storage_dir}")
    print(f"  扩展名: {file_exts}")
    print(f"  提示: 启动后用 ingest <路径> 添加文档")
    return kb_id, cfg


def _select_single_kb(configs: dict) -> tuple:
    """启动前选择单一知识库。返回 (kb_id, cfg)。
    不同知识库彼此独立，避免内容污染。
    一直回车 = DEFAULT_KB_ID（找不到则第 1 个）。
    始终显示菜单（即使只有一个），并支持新建数据集。
    """
    if not configs:
        # 无配置：直接进入新建流程
        print(f"[知识库] 未找到 {KB_CONFIGS_DIR}/*.yaml")
        print("[知识库] 请新建第一个数据集：")
        return _create_new_kb(configs)

    kb_ids = list(configs.keys())
    # 默认选中 DEFAULT_KB_ID
    default_idx = 1
    for i, kid in enumerate(kb_ids, 1):
        if kid == DEFAULT_KB_ID:
            default_idx = i
            break

    # 始终显示菜单（即使只有一个），让用户明确当前在哪个库
    # 最后一项是"新建数据库"
    new_kb_idx = len(kb_ids) + 1
    print("\n" + "=" * 60)
    print(" 请选择要进入的数据库（不同数据库彼此独立，避免内容污染）")
    print("=" * 60)
    for i, kid in enumerate(kb_ids, 1):
        cfg = configs[kid]
        marker = " (默认)" if i == default_idx else ""
        storage_built = Path(cfg.get("storage_dir", ""), "docstore.json").exists()
        if storage_built:
            size = format_size(get_storage_size(cfg.get("storage_dir", "")))
            built_tag = f" [已有数据 {size}]"
        else:
            built_tag = " [空库]"
        print(f"  {i}. {cfg.get('name', kid)}{marker}  [{kid}]{built_tag}")
        print(f"     存储到: {cfg.get('storage_dir', '?')}")
    print(f"  {new_kb_idx}. + 新建数据库")
    print("=" * 60)
    while True:
        try:
            choice = input(f"输入序号 (1-{new_kb_idx})，默认 {default_idx}: ").strip() or str(default_idx)
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= new_kb_idx:
            if int(choice) == new_kb_idx:
                # 新建数据集
                new_kb_id, new_cfg = _create_new_kb(configs)
                configs[new_kb_id] = new_cfg  # 加入 configs 供 kbs 命令查看
                return new_kb_id, new_cfg
            kb_id = kb_ids[int(choice) - 1]
            return kb_id, configs[kb_id]
        print(f"无效输入: {choice}，请输入 1-{new_kb_idx} 的数字")


def main() -> None:
    # 1) 选择知识库（必须在系统启动前完成，不同知识库彼此独立，避免内容污染）
    configs = _load_kb_configs()
    # 兼容旧的单库模式（无配置时）
    if not configs:
        configs = {"_default": {
            "name": "默认",
            "description": "默认数据库",
            "storage_dir": STORAGE_DIR,
            "file_exts": [".md"],
        }}
    current_kb_id, current_cfg = _select_single_kb(configs)

    # 2) 选择并初始化 LLM
    provider = _select_llm_provider()
    Settings.llm = _build_llm(provider)
    provider_names = {
        "deepseek": "DeepSeek",
        "qwen": "通义千问",
        "zhipu": "智谱GLM",
        "ollama": "Ollama",
        "custom": "自定义OpenAI兼容",
    }
    print(f"[LLM] 已初始化: {provider_names.get(provider, provider)}")

    # 3) 选择并初始化 embedding 模型
    global EMBED_MODEL
    EMBED_MODEL = _select_embed_model()
    Settings.embed_model = _build_embed_model(EMBED_MODEL)
    embed_desc = "Qwen2-7B(FP16)" if _is_qwen2_embed(EMBED_MODEL) else "小模型(FP16)"
    print(f"[Embedding] 已加载: {EMBED_MODEL} ({embed_desc})")

    # 4) 加载/构建当前知识库的索引
    print("\n" + "=" * 60)
    print(f" LlamaIndex 交互式 RAG 问答 ({provider_names.get(provider, provider)} + {embed_desc})")
    print("=" * 60)
    print(f"[加载知识库] {current_cfg.get('name', current_kb_id)} [{current_kb_id}]")
    print(f"  存储到: {current_cfg['storage_dir']}")
    index = _load_or_build_kb(current_kb_id, current_cfg)
    sys.stdout.flush()  # 确保索引加载完成后终端输出完整
    query_engine = _make_query_engine(index)

    # 5) 选择检索模式
    current_mode = _select_retrieval_mode()
    print(f"[检索模式] 当前: {RETRIEVAL_MODES[current_mode]}")

    # 6) 显示当前会话信息
    kb_size = get_storage_size(current_cfg["storage_dir"])
    print("\n" + "=" * 60)
    print(f" 当前知识库: {current_cfg.get('name', current_kb_id)} [{current_kb_id}]")
    print(f" 存储位置:   {current_cfg['storage_dir']} ({format_size(kb_size)})")
    print(f" 检索模式:   {RETRIEVAL_MODES[current_mode]}")
    _sh_status = "开" if _RETR_CFG["dedup"]["simhash_enabled"] else "关"
    print(f" 去重: 哈希去重 (DUPLICATES_ONLY) + SimHash 转载拦截（{_sh_status}）")
    print(" 提示: 使用 ingest <路径> 添加文档到当前数据库")
    print("=" * 60)
    print("索引就绪。输入问题即可获取回答。")
    print("命令：  exit / quit / 退出  -> 结束；  clear  -> 清屏")
    print("      mode  -> 切换检索模式（向量检索/聚合直遍/全文搜索）")
    print("      kbs  -> 查看所有可用知识库（切换需重启程序）")
    print("      rebuild  -> 重建当前知识库索引")
    print("      embed <路径>  -> 向量化单个文件并打印结果（不写入主索引）")
    print("      ingest <路径>  -> 把指定文件/文件夹增量加入当前知识库（哈希+转载去重）")
    print("      list  -> 列出当前知识库中所有已 ingest 的文件路径")
    print("      dedup_status  -> 查看去重统计（哈希数/文档数/转载拦截数）")
    print("-" * 60)
    sys.stdout.flush()  # 确保终端输出完整，避免 input() 卡顿

    # ANSI 配色：红底亮白字加粗（高对比度，避免红+蓝混合产生的紫色感）
    # 说明：红字+蓝底(31;44)在终端中视觉上会混合成紫色，对比度低；
    #       改用红底(41)+亮白字(97)+加粗(1)，红色背景形成明显色块，
    #       在庞大文本中最易定位，白字在红底上对比度最高，不产生紫色感。
    QUESTION_HIGHLIGHT = "\033[1;97;41m"
    RESET = "\033[0m"

    # 模式简称（显示在提示符中）
    MODE_SHORT = {"vector": "向量", "aggregate": "聚合", "fulltext": "全文"}

    while True:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            # 提示符显示当前检索模式
            mode_tag = MODE_SHORT.get(current_mode, "?")
            question = input(f"\n{QUESTION_HIGHLIGHT}[{mode_tag}] 你的问题> ").strip()
            sys.stdout.write(RESET)
            sys.stdout.flush()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "退出", "q"}:
            print("再见！")
            break
        if question.lower() in {"clear", "cls"}:
            os.system("cls" if os.name == "nt" else "clear")
            continue

        # mode：切换检索模式
        if question.lower() in {"mode", "模式", "检索模式"}:
            print("\n" + "=" * 60)
            print(" 切换检索模式（当前: " + RETRIEVAL_MODES[current_mode] + "）")
            print("=" * 60)
            mode_keys = list(RETRIEVAL_MODES.keys())
            for i, key in enumerate(mode_keys, 1):
                marker = " [当前]" if key == current_mode else ""
                print(f"  {i}. {RETRIEVAL_MODES[key]}{marker}")
            # 显示全文搜索索引状态
            if _fulltext_searcher is not None:
                print(f"     全文搜索索引: {_fulltext_searcher.status}")
            print("-" * 60)
            try:
                choice = input(f"输入序号 (1-{len(mode_keys)})，回车取消: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("已取消")
                continue
            if choice.isdigit() and 1 <= int(choice) <= len(mode_keys):
                current_mode = mode_keys[int(choice) - 1]
                print(f"  -> 已切换到: {RETRIEVAL_MODES[current_mode]}")
                sys.stdout.flush()
            continue

        # kbs：查看所有可用知识库（切换需重启）
        if question.lower() in {"kbs", "知识库", "kb"}:
            print("\n" + "=" * 60)
            print(" 所有可用数据库（切换需重启程序）")
            print("=" * 60)
            all_configs = _load_kb_configs()
            for kid, cfg in all_configs.items():
                current_mark = " [当前]" if kid == current_kb_id else ""
                exists = Path(cfg["storage_dir"]).exists() and (Path(cfg["storage_dir"]) / "docstore.json").exists()
                if exists:
                    size = format_size(get_storage_size(cfg["storage_dir"]))
                    # 尝试获取节点数
                    try:
                        sc = StorageContext.from_defaults(persist_dir=cfg["storage_dir"])
                        node_count = len(sc.docstore.docs)
                    except Exception:
                        node_count = "?"
                else:
                    size = "空"
                    node_count = 0
                print(f"  - {cfg.get('name', kid)} [{kid}]{current_mark}")
                print(f"      存储:   {size}  ({cfg['storage_dir']})")
                print(f"      节点数: {node_count}")
            print("-" * 60)
            print(f"  当前会话: {current_cfg.get('name', current_kb_id)} [{current_kb_id}]")
            print("  切换数据库请退出后重新启动程序")
            print("=" * 60)
            continue

        # rebuild：清空当前数据库，重建空索引
        if question.lower() == "rebuild":
            import shutil
            kb_storage = current_cfg["storage_dir"]
            size_before = get_storage_size(kb_storage)
            if Path(kb_storage).exists():
                shutil.rmtree(kb_storage)
                print(f"已清空数据库（原 {format_size(size_before)}），重建空索引 ...")
            else:
                print("数据库不存在，创建空索引 ...")
            index = _load_or_build_kb(current_kb_id, current_cfg)
            query_engine = _make_query_engine(index)
            print(f"重建完成: {current_cfg.get('name', current_kb_id)}（空库，请用 ingest 添加文档）")
            sys.stdout.flush()
            continue

        # dedup_status：查看当前知识库的去重统计
        if question.lower() in {"dedup_status", "dedup", "去重统计"}:
            print("\n" + "=" * 60)
            print(f"去重统计 (IngestionPipeline + DUPLICATES_ONLY)")
            print(f"知识库: {current_cfg.get('name', current_kb_id)} [{current_kb_id}]")
            print("=" * 60)
            try:
                docstore = index.storage_context.docstore
                all_hashes = docstore.get_all_document_hashes()
                docs = docstore.docs
                source_files = set()
                for _, node in docs.items():
                    fp = (node.metadata or {}).get("file_path", "未知")
                    source_files.add(fp)
                try:
                    vec_count = len(index.vector_store.data.embedding_dict)
                except Exception:
                    vec_count = "?"
                print(f"  docstore 跟踪的唯一哈希数: {len(all_hashes)}")
                print(f"  docstore 存储的文档/节点数: {len(docs)}")
                print(f"  不同来源文件数:           {len(source_files)}")
                print(f"  向量库中的向量数:          {vec_count}")
                print(f"  存储目录总容量:            {format_size(get_storage_size(current_cfg['storage_dir']))}")
                # SimHash 转载拦截统计（指纹库存在才显示）
                fp_file = Path(current_cfg["storage_dir"]) / _SIMHASH_FILE
                if fp_file.exists():
                    sh_store = _SimHashStore(fp_file)
                    sh_store.load()
                    print("-" * 60)
                    print(f"  SimHash 内容指纹数:        {sh_store.count}")
                    print(f"  累计拦截转载:              {sh_store.total_dups}")
                print("-" * 60)
                print("说明：")
                print("  - 唯一哈希 = 曾 ingest 过的去重后文档数")
                print("  - 重复文档在 ingest 时被自动跳过，不会出现在上述计数中")
                print("  - 转载拦截 = SimHash 内容指纹判定的近似转载（含不同文件的同文转载）")
                print("  - 跳过数 = 历次 ingest 时输入文档数 - 新增哈希数（不累计）")
                print("=" * 60)
            except Exception as e:
                print(f"[dedup_status 出错] {type(e).__name__}: {e}")
            continue

        # embed <文件路径>：向量化单个文档并打印结果（不写入主索引）
        if question.lower().startswith("embed ") or question.lower().startswith("向量化 "):
            file_path = question.split(" ", 1)[1].strip().strip('"').strip("'")
            if not Path(file_path).exists():
                print(f"[错误] 文件不存在: {file_path}")
                continue
            print(f"\n[向量化] 读取文件: {file_path}")
            try:
                docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                print(f"    -> 加载 {len(docs)} 个文档")
                nodes = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).get_nodes_from_documents(docs, show_progress=True)
                print(f"    -> 切成 {len(nodes)} 个 chunk，开始向量化（本地 embedding）...")
                texts = [n.text for n in nodes]
                vectors = Settings.embed_model.get_text_embedding_batch(texts, show_progress=True)
                parts = [
                    "",
                    "=" * 60,
                    f"向量化完成！共 {len(vectors)} 个向量，维度 = {len(vectors[0])}",
                    "=" * 60,
                ]
                for i, (node, vec) in enumerate(zip(nodes, vectors), 1):
                    preview = node.text.strip().replace("\n", " ")[:80]
                    parts.append(f"  chunk {i:>2} | 前3维={[round(x,4) for x in vec[:3]]} | 内容预览: {preview}...")
                parts.append("=" * 60)
                print_paged("\n".join(parts))
            except Exception as e:
                print(f"[向量化出错] {type(e).__name__}: {e}")
            sys.stdout.flush()  # 确保终端输出完整
            continue

        # ingest <路径>：增量加入当前知识库
        if question.lower().startswith("ingest ") or question.lower().startswith("加入 "):
            target = question.split(" ", 1)[1].strip().strip('"').strip("'")
            path = Path(target)
            if not path.exists():
                print(f"[错误] 路径不存在: {target}")
                continue
            # 校验模型一致性
            if not _check_embed_consistency(index.storage_context):
                print("[已拦截] 请先执行 rebuild 重建索引，再 ingest 新文档。")
                continue
            print(f"\n[ingest] 目标: {path} -> 知识库 [{current_cfg.get('name', current_kb_id)}]")
            try:
                # 惰性读取：逐文件加载，避免一次性把整个语料库载入内存
                # （旧版 load_data() 会先读入全部文档，大语料下内存随文件数线性膨胀）
                if path.is_file():
                    reader = SimpleDirectoryReader(input_files=[str(path)], filename_as_id=True)
                    total_files = 1
                else:
                    exts = current_cfg.get("file_exts", [".md", ".txt"])
                    reader = SimpleDirectoryReader(input_dir=str(path), required_exts=exts, recursive=True, filename_as_id=True)
                    exts_l = {e.lower() for e in exts}
                    total_files = sum(1 for p in Path(path).rglob("*") if p.is_file() and p.suffix.lower() in exts_l)
                # 分批参数：从 retrieval_config.yaml 读取（可在配置文件中调整）
                batch_size = _RETR_CFG["ingest"]["batch_size"]
                auto_continue_timeout = _RETR_CFG["ingest"]["auto_continue_timeout"]
                # 单文件路径时强制单批（避免无意义的暂停提示）
                if path.is_file():
                    batch_size = max(batch_size, total_files)
                print(f"    -> 发现 {total_files} 个文件，分批处理（每批 {batch_size} 个），逐文件加载并增量写入（哈希+转载去重）...")
                pipeline = make_pipeline(index.storage_context)
                size_before = get_storage_size(current_cfg["storage_dir"])

                # SimHash 近似转载拦截（报纸转载内容近乎相同，哈希去重拦不住，见 _SimHashStore 注释）
                simhash_store = _get_simhash_store(current_cfg["storage_dir"], docstore=index.storage_context.docstore)
                near_skipped = 0
                batch_near_skipped = 0

                # 提速关键：缓存全库哈希。llama-index 的 DUPLICATES_ONLY 在每次
                # pipeline.run 时都全量重建哈希字典（O(已存条目数)），逐文档调用
                # 总开销 O(N²)——数千文件后每个文件仅哈希扫描就要几百毫秒且越来越慢。
                # 这里缓存一份并在 set_document_hash 时增量更新，使每次检查降为 O(1)。
                # 注：切块节点的哈希仍写入真实 docstore（不参与输入文档级去重判断），
                # 文档级去重语义与原版完全一致。
                docstore = index.storage_context.docstore
                _orig_get_hashes = docstore.get_all_document_hashes
                _orig_set_hash = docstore.set_document_hash
                _hash_cache = _orig_get_hashes()

                def _get_hashes_cached():
                    return _hash_cache

                def _set_hash_cached(doc_id, doc_hash):
                    _orig_set_hash(doc_id, doc_hash)
                    _hash_cache[doc_hash] = doc_id

                object.__setattr__(docstore, "get_all_document_hashes", _get_hashes_cached)
                object.__setattr__(docstore, "set_document_hash", _set_hash_cached)

                all_new_nodes = []
                new_docs = 0
                skipped = 0
                total_nodes = 0
                # 批次内统计（用于批次结束时的提示）
                batch_new_docs = 0
                batch_skipped = 0
                batch_nodes = 0
                batch_num = 0
                user_aborted = False
                i = 0
                try:
                    for doc_batch in tqdm(reader.iter_data(), desc="ingest", unit="file", total=total_files, ncols=80, leave=False):
                        for doc in doc_batch:
                            i += 1
                            fname = (doc.metadata or {}).get("file_name", "") or doc.get_text()[:30]
                            # SimHash 转载拦截：与库内已有文章内容近似（含归一化后精确一致）则跳过
                            dup_key, dup_dist, dup_kept = simhash_store.check_text(doc.get_text()) if simhash_store is not None else (None, -1, "")
                            if dup_key is not None:
                                near_skipped += 1
                                batch_near_skipped += 1
                                simhash_store.record_dup(dup_key, (doc.metadata or {}).get("file_path", ""))
                                tqdm.write(f"    [{i}/{total_files}] 跳过转载: {fname}（与 {Path(dup_kept).name or dup_kept} 内容重复，汉明距离={dup_dist}）")
                            else:
                                new_nodes = pipeline.run(documents=[doc], show_progress=False)
                                all_new_nodes.extend(new_nodes)
                                total_nodes += len(new_nodes)
                                batch_nodes += len(new_nodes)
                                if new_nodes:
                                    new_docs += 1
                                    batch_new_docs += 1
                                    if simhash_store is not None:
                                        simhash_store.add_text(doc.get_text(), (doc.metadata or {}).get("file_path", ""))
                                else:
                                    skipped += 1
                                    batch_skipped += 1
                                tqdm.write(f"    [{i}/{total_files}] 处理: {fname}（产出 {len(new_nodes)} 节点）")
                            # 批次边界：达到 batch_size 或最后一批
                            # 持久化 + 交互式暂停（10秒超时自动继续）
                            if i % batch_size == 0 or i >= total_files:
                                is_last_batch = i >= total_files
                                # 写索引 + 流式持久化（中断后重跑 ingest 会因哈希去重自动跳过已落盘的文件）
                                if all_new_nodes:
                                    index.insert_nodes(all_new_nodes)
                                    all_new_nodes.clear()
                                index.storage_context.persist(persist_dir=current_cfg["storage_dir"])
                                if simhash_store is not None:
                                    simhash_store.save()
                                if is_last_batch:
                                    tqdm.write(f"    [批次 {batch_num + 1}] 前 {i}/{total_files} 个文件已落盘（最后一批）")
                                    break
                                # 中间批次：提示用户继续或退出
                                batch_num += 1
                                tqdm.write(
                                    f"    [批次 {batch_num}] 已处理 {i}/{total_files} 个文件 "
                                    f"（本批新增 {batch_new_docs}，跳过 {batch_skipped}，转载 {batch_near_skipped}，"
                                    f"产出 {batch_nodes} 节点），剩余 {total_files - i} 个"
                                )
                                prompt = (
                                    f"    按回车继续下一批，输入 q 退出 "
                                    f"（{auto_continue_timeout}秒后自动继续）: "
                                )
                                choice = _input_with_timeout(prompt, auto_continue_timeout)
                                if choice.strip().lower() in {"q", "quit", "退出"}:
                                    print(f"    [已中断] 用户选择退出，已处理 {i}/{total_files} 个文件已落盘")
                                    user_aborted = True
                                    break
                                # 重置批次统计
                                batch_new_docs = 0
                                batch_skipped = 0
                                batch_nodes = 0
                                batch_near_skipped = 0
                        if user_aborted or i >= total_files:
                            break
                finally:
                    object.__setattr__(docstore, "get_all_document_hashes", _orig_get_hashes)
                    object.__setattr__(docstore, "set_document_hash", _orig_set_hash)
                # 最终持久化（处理未达批次边界的剩余节点 + 用户中途退出场景）
                if all_new_nodes:
                    index.insert_nodes(all_new_nodes)
                    all_new_nodes.clear()
                index.storage_context.persist(persist_dir=current_cfg["storage_dir"])
                if simhash_store is not None:
                    simhash_store.save()
                hashes_after = len(index.storage_context.docstore.get_all_document_hashes())
                size_after = get_storage_size(current_cfg["storage_dir"])
                size_delta = size_after - size_before
                if user_aborted:
                    print(f"    -> 已中止！本次处理 {i}/{total_files} 个文件，新增 {new_docs} 个文档（{total_nodes} 个节点），跳过 {skipped} 个哈希重复、{near_skipped} 篇转载")
                    print(f"    -> 剩余 {total_files - i} 个文件未处理，下次重跑 ingest 会因去重自动跳过已落盘文件")
                else:
                    print(f"    -> 完成！新增 {new_docs} 个文档（{total_nodes} 个节点），跳过 {skipped} 个哈希重复、{near_skipped} 篇转载")
                print(f"    -> 已持久化到 {current_cfg['storage_dir']}，docstore 共跟踪 {hashes_after} 个唯一哈希")
                print(f"    -> 存储容量: {format_size(size_after)}（本次 {'+' if size_delta >= 0 else ''}{format_size(size_delta)}）")
                sys.stdout.flush()  # 确保终端输出完整，避免后续 input() 卡顿
                query_engine = _make_query_engine(index)
            except Exception as e:
                print(f"[ingest 出错] {type(e).__name__}: {e}")
            sys.stdout.flush()  # 确保终端输出完整
            continue

        # list：列出当前知识库中已 ingest 的文件路径
        if question.lower() in {"list", "列表", "ls"}:
            try:
                docstore = index.storage_context.docstore
                seen = set()
                file_list = []
                for _, node in docstore.docs.items():
                    fp = (node.metadata or {}).get("file_path", "未知来源")
                    if fp not in seen:
                        seen.add(fp)
                        file_list.append(fp)
                try:
                    vec_count = len(index.vector_store.data.embedding_dict)
                except Exception:
                    vec_count = "?"
                parts = [
                    "",
                    "=" * 60,
                    f"当前知识库 [{current_cfg.get('name', current_kb_id)}] 已 ingest 的文件路径：",
                    "=" * 60,
                ]
                for i, fp in enumerate(file_list, 1):
                    parts.append(f"  {i:>3}. {fp}")
                parts.append("=" * 60)
                parts.append(f"共 {len(file_list)} 个不同来源文件（{len(docstore.docs)} 个文档/节点，{vec_count} 个向量）")
                parts.append("=" * 60)
                print_paged("\n".join(parts))
            except Exception as e:
                print(f"[list 出错] {type(e).__name__}: {e}")
            continue

        # 普通提问（根据当前检索模式分派）
        try:
            # ---- 模式1：聚合直遍 ----
            if current_mode == "aggregate":
                _handle_aggregate_query(question, index)
                continue

            # ---- 模式2：全文搜索 ----
            if current_mode == "fulltext":
                _handle_fulltext_query(question, index)
                continue

            # ---- 模式3：向量检索（默认）----
            # 向量模式下，如果是聚合类查询，自动提示可切换模式
            if _is_aggregate_query(question):
                print(f"\n[提示] 检测到聚合查询，当前为向量检索模式（仅返回 top-k 片段，无法列出全部）。")
                print(f"       输入 'mode' 切换到「聚合直遍」模式可获取全部结果。")
                print(f"       本次仍用向量检索继续...")

            response = query_engine.query(question)
            print("\n回答:")
            print_paged(str(response))

            # 打印检索到的笔记原文片段
            if getattr(response, "source_nodes", None):
                parts = [
                    "",
                    "=" * 60,
                    f"检索到 {len(response.source_nodes)} 个相关片段 (来自: {current_cfg.get('name', current_kb_id)}):",
                    "=" * 60,
                ]
                for i, node in enumerate(response.source_nodes, 1):
                    meta = node.node.metadata or {}
                    file_path = meta.get("file_path", "未知来源")
                    parts.append(f"\n--- 片段 {i} | 来源: {file_path} ---")
                    highlighted = _highlight_relevant_sentences(node.node.get_content(), question)
                    parts.append(highlighted)
                parts.append("\n" + "=" * 60)
                print_paged("\n".join(parts))
            else:
                print("\n（未检索到 source_nodes）")
        except Exception as e:
            print(f"\n[查询出错] {type(e).__name__}: {e}")
            print("可继续输入下一个问题。")


if __name__ == "__main__":
    main()
