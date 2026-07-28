# 项目文件结构解析与功能摘要

> **同步更新约定**：本文档随 `run_vector_demo.py` 的改动同步更新。
> 更新检查点：新增/删除函数、新增命令、修改入口流程、调整配置文件格式、新增依赖包。
> 快速校验命令：`.venv\Scripts\python.exe -c "import ast; tree=ast.parse(open('run_vector_demo.py',encoding='utf-8').read()); print([(n.lineno,n.name) for n in tree.body if isinstance(n, ast.FunctionDef)])"`
> 最近同步：2026-07-28（合并 stash 补丁后全面刷新，行号以实测为准）

***

## 1. 文件结构

```
c:\code\LlamaIndex\
│
├── run_vector_demo.py        # CLI 主程序（1962 行，入口 + REPL + 所有业务逻辑）
├── app.py                    # Streamlit Web UI（737 行，import 复用 run_vector_demo，不重写业务逻辑）
├── retrieval_config.yaml     # 检索参数统一配置（CLI/Web 共用，缺失时用代码内默认值）
├── MANUAL.md                 # 用户使用手册
├── requirements.txt          # 项目依赖清单
├── highlight_config.json     # 高亮样式配置（颜色/下划线/加粗等）
│
├── kb_configs/               # 知识库配置目录（每个 .yaml = 一个独立知识库）
│   ├── code.yaml                  # 代码库
│   ├── knowledge.yaml             # 知识库
│   ├── newspaper.yaml             # 报纸库（已构建索引）
│   └── *.yaml.example             # 配置模板（beijing_daily / history / literature）
│
├── models/                   # 本地 embedding 模型目录（便于离线/便携部署）
│   └── bge-m3/                    # BAAI/bge-m3（1024 维，FP16/GPU）
│
├── storage/                  # 索引持久化目录（每个知识库独立子目录）
│   └── newspaper/                 # 报纸库索引
│       ├── docstore.json              # 文档存储（含原文+哈希）
│       ├── index_store.json           # 索引结构
│       └── default__vector_store.json # 向量存储
│
└── _deploy_pkg/              # 部署脚本（deploy.ps1，torch 版本已过期，勿直接重跑，见 MANUAL.md）
```

**核心文件**（本项目自研）：

- `run_vector_demo.py` —— CLI 主程序，所有业务逻辑都在这里
- `app.py` —— Web UI，通过 `from run_vector_demo import (...)` 复用全部业务函数
- `kb_configs/*.yaml` —— 知识库配置
- `retrieval_config.yaml` —— 检索参数配置
- `highlight_config.json` —— 展示层配置
- `models/` —— 本地模型文件
- `storage/` —— 运行时生成的索引数据

**测试/工具脚本**：`test_topk.py`（检索精度对比）、`test_qwen2_embed.py`（模型验证）、`test_modes.py`（检索模式测试）、`analyze_newspapers.py`（语料分析）、`generate_kb_configs.py`（批量生成库配置）、`download_models.py`（模型下载）

***

## 2. 包引用关系

### 2.1 第三方依赖

```python
# 核心框架
from llama_index.core import (
    VectorStoreIndex,           # 向量索引
    SimpleDirectoryReader,      # 目录读取器
    load_index_from_storage,    # 从存储加载索引
    StorageContext,             # 存储上下文
    Settings,                   # 全局配置
)
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy  # 去重管道
from llama_index.core.node_parser import SentenceSplitter                    # 文本切块

# LLM 提供商（按需导入，可选）
from llama_index.llms.deepseek import DeepSeek              # DeepSeek
from llama_index.llms.openai_like import OpenAILike         # 通义千问/智谱GLM/自定义
from llama_index.llms.ollama import Ollama                  # Ollama 本地模型

# Embedding 模型
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 辅助库
import torch               # PyTorch（FP16 加载、GPU 推理）
import numpy as np         # 向量运算（余弦相似度）
from tqdm import tqdm      # 进度条
import yaml                # 知识库/检索配置解析（PyYAML）
```

`app.py` 额外依赖 `streamlit`。

### 2.2 依赖关系图

```
run_vector_demo.py ◄── app.py（import 复用 18 个函数/对象，含流式持久化补丁自动生效）
    │
    ├── llama_index.core ── VectorStoreIndex, IngestionPipeline, SentenceSplitter
    │       └── 提供：索引构建、去重、切块、查询引擎
    │
    ├── llama_index.llms.* ── DeepSeek / OpenAILike / Ollama
    │       └── 提供：LLM 推理（按用户选择加载其中一个）
    │
    ├── llama_index.embeddings.huggingface ── HuggingFaceEmbedding
    │       └── 提供：文本向量化（本地模型，必选）
    │
    └── torch / numpy / tqdm / yaml
            └── 提供：GPU 推理、向量运算、进度显示、配置解析
```

### 2.3 环境变量依赖

| 环境变量                | 用途                   | 必填             |
| ------------------- | -------------------- | -------------- |
| `DEEPSEEK_API_KEY`  | DeepSeek API 密钥      | 选 DeepSeek 时必填 |
| `DASHSCOPE_API_KEY` | 阿里云通义千问密钥            | 选 Qwen 时必填     |
| `ZHIPU_API_KEY`     | 智谱 GLM 密钥            | 选 GLM 时必填      |
| `DATA_DIR`          | 数据源目录（覆盖默认）          | 否              |
| `STORAGE_DIR`       | 存储目录（覆盖默认）           | 否              |
| `EMBED_MODEL`       | embedding 模型路径（跳过菜单） | 否              |
| `LLM_PROVIDER`      | LLM 提供商（跳过菜单）        | 否              |
| `MAX_DOCS`          | 限制文档数（调试用）           | 否              |

***

## 3. 入口函数

### 3.1 主入口

```python
# run_vector_demo.py 末尾
if __name__ == "__main__":
    main()
```

### 3.2 `main()` 函数（第 1513 行）

主流程的 6 个阶段：

```
main()
  │
  ├─ 阶段1: 选择知识库
  │   ├─ _load_kb_configs()          # 扫描 kb_configs/*.yaml
  │   └─ _select_single_kb(configs)  # 交互式菜单（始终显示，支持新建）
  │
  ├─ 阶段2: 选择并初始化 LLM
  │   ├─ _select_llm_provider()      # 交互式选择 5 种提供商
  │   └─ _build_llm(provider)        # 构建 LLM 实例
  │
  ├─ 阶段3: 选择并初始化 embedding 模型
  │   ├─ _select_embed_model()       # 扫描本地模型 + 交互选择
  │   └─ _build_embed_model(path)    # 构建 embedding 实例（自动适配 Qwen2）
  │
  ├─ 阶段4: 加载/构建索引
  │   └─ _load_or_build_kb(kb_id, cfg)  # 加载已有索引 或 从数据源构建
  │
  ├─ 阶段5: 选择检索模式
  │   └─ _select_retrieval_mode()    # 向量检索 / 聚合直遍 / 全文搜索
  │
  └─ 阶段6: REPL 交互循环
      └─ while True: input() → 命令分发 → 查询/继续
```

Web 入口：`streamlit run app.py`（复用同一套函数，无独立业务逻辑）。

***

## 4. 调用的函数

### 4.1 函数清单（34 个，按功能分组）

#### 检索参数配置（2 个）

| 函数                         | 行号 | 功能                                        |
| -------------------------- | -- | ----------------------------------------- |
| `_load_retrieval_config()` | 59 | 加载 retrieval\_config.yaml，缺失/出错时用代码内默认值    |
| `_make_query_engine(index)` | 88 | 统一创建 query\_engine，参数取自配置（CLI/Web 共用）     |

#### 持久化补丁（1 个）

| 函数                            | 行号  | 功能                                                              |
| ----------------------------- | --- | --------------------------------------------------------------- |
| `_streaming_kvstore_persist()` | 106 | monkey-patch `SimpleKVStore.persist`：`json.dump` 流式写盘，防大库持久化内存翻倍 |

#### LLM 管理（2 个）

| 函数                       | 行号  | 功能                 |
| ------------------------ | --- | ------------------ |
| `_select_llm_provider()` | 141 | 交互式选择 LLM 提供商（5 种） |
| `_build_llm(provider)`   | 170 | 根据提供商构建 LLM 实例     |

#### Embedding 管理（4 个）

| 函数                           | 行号  | 功能                             |
| ---------------------------- | --- | ------------------------------ |
| `_scan_local_embed_models()` | 247 | 扫描本地模型目录 + HF 缓存               |
| `_is_qwen2_embed(path)`      | 282 | 判断是否是 Qwen2 系列 embedding       |
| `_select_embed_model()`      | 286 | 交互式选择 embedding 模型             |
| `_build_embed_model(path)`   | 327 | 构建 embedding 实例（自动适配 Qwen2 配置） |

#### 索引/存储（6 个）

| 函数                                                | 行号        | 功能                         |
| ------------------------------------------------- | --------- | -------------------------- |
| `_get_embed_dim()`                                | 358       | 获取当前 embedding 输出维度        |
| `_check_embed_consistency(storage_context)`       | 363       | 校验索引向量维度与当前模型一致            |
| `get_storage_size(path)` / `format_size(bytes)`   | 392 / 407 | 存储容量计算与格式化                 |
| `make_pipeline(storage_context)`                  | 1051      | 创建带哈希去重的 IngestionPipeline |
| `_read_documents_with_progress(reader, data_dir)` | 1070      | 带实时进度条的文档读取                |

#### 展示层（4 个）

| 函数                                           | 行号  | 功能             |
| -------------------------------------------- | --- | -------------- |
| `_load_highlight_config()`                   | 430 | 加载高亮样式配置       |
| `_build_ansi_code(cfg)`                      | 466 | 构建 ANSI 转义码    |
| `_highlight_relevant_sentences(text, query)` | 498 | 高亮片段中与查询最相似的句子 |
| `print_paged(text, page_lines)`              | 583 | 分页显示长文本        |

#### 输入工具（1 个）

| 函数                              | 行号  | 功能                                       |
| ------------------------------- | --- | ---------------------------------------- |
| `_input_with_timeout(prompt, timeout)` | 629 | 带超时的 input（Windows msvcrt / Unix select），ingest 批次间确认用 |

#### 聚合查询（3 个）

| 函数                                         | 行号  | 功能                    |
| ------------------------------------------ | --- | --------------------- |
| `_is_aggregate_query(question)`            | 711 | 检测是否是聚合查询（"列出所有X"类）   |
| `_extract_keyword_from_query(question)`    | 719 | 从聚合查询提取关键词            |
| `_handle_aggregate_query(question, index)` | 729 | 直遍 docstore 提取，跳过向量检索 |

#### 全文检索（3 个）

| 函数                              | 行号   | 功能                          |
| ------------------------------- | ---- | --------------------------- |
| `_get_fulltext_searcher(index)` | 948  | 构建/复用 BM25 全文检索器（中文 bigram） |
| `_handle_fulltext_query(...)`   | 956  | 全文搜索查询处理（不调 LLM）            |
| `_highlight_keywords(text, ...)` | 1012 | 搜索结果关键词高亮                   |

#### 检索模式（1 个）

| 函数                         | 行号   | 功能                        |
| -------------------------- | ---- | ------------------------- |
| `_select_retrieval_mode()` | 1032 | 启动时选择检索模式，运行中 `mode` 命令切换 |

#### 知识库管理（6 个）

| 函数                              | 行号   | 功能                             |
| ------------------------------- | ---- | ------------------------------ |
| `_load_kb_configs()`            | 1162 | 扫描 kb\_configs/ 目录加载所有 yaml 配置 |
| `_select_knowledge_base()`      | 1188 | 旧版知识库选择（已弃用，保留兼容）              |
| `_load_or_build_kb(kb_id, cfg)` | 1240 | 加载或构建指定知识库的索引                  |
| `load_or_build_index()`         | 1326 | 旧版单库模式（保留兼容）                   |
| `_create_new_kb(configs)`       | 1372 | 交互式新建知识库（保存为 yaml）             |
| `_select_single_kb(configs)`    | 1458 | **当前使用**的启动前知识库选择菜单            |

#### 主流程（1 个）

| 函数       | 行号   | 功能          |
| -------- | ---- | ----------- |
| `main()` | 1513 | 入口函数，协调所有模块 |

### 4.2 测试脚本

| 文件                    | 功能                                     |
| --------------------- | -------------------------------------- |
| `test_topk.py`        | 对比不同 `similarity_top_k` 的检索效果 + 测试聚合查询 |
| `test_qwen2_embed.py` | 验证 gte-Qwen2-7B 模型加载和维度                |
| `test_modes.py`       | 检索模式（向量/聚合/全文）测试                       |

***

## 5. 项目调用链条

### 5.1 启动调用链

```
python run_vector_demo.py
    │
    └─ main()
        │
        ├─[1] _load_kb_configs()
        │       └─ 扫描 kb_configs/*.yaml → 返回 {kb_id: cfg}
        │
        ├─[2] _select_single_kb(configs)
        │       ├─ 显示菜单（始终显示，即使只有一个）
        │       ├─ 用户选择 → 返回 (kb_id, cfg)
        │       └─ 若选"新建" → _create_new_kb() → 保存 yaml
        │
        ├─[3] _select_llm_provider()
        │       └─ 用户选择 → 返回 provider 字符串
        ├─[4] _build_llm(provider)
        │       └─ 读取环境变量 API_KEY → 构建 LLM 实例
        │
        ├─[5] _select_embed_model()
        │       ├─ _scan_local_embed_models() → 扫描 models/ + HF 缓存
        │       └─ 用户选择 → 返回模型路径
        ├─[6] _build_embed_model(path)
        │       ├─ _is_qwen2_embed(path) → 判断是否 Qwen2
        │       └─ 构建 HuggingFaceEmbedding（Qwen2 加 instruction）
        │
        ├─[7] _load_or_build_kb(kb_id, cfg)
        │       │
        │       ├─[分支A] 已有索引（docstore.json 存在）
        │       │   ├─ 后台线程显示进度条（基于存储大小预估）
        │       │   ├─ StorageContext.from_defaults(persist_dir)
        │       │   ├─ _check_embed_consistency(storage_context)
        │       │   └─ load_index_from_storage() → 返回 index
        │       │
        │       └─[分支B] 无索引，从数据源构建
        │           ├─ SimpleDirectoryReader(input_dir, required_exts)
        │           ├─ _read_documents_with_progress(reader, data_dir)
        │           │   └─ ANSI Live 显示读取进度
        │           ├─ make_pipeline(storage_context)
        │           │   └─ IngestionPipeline(SentenceSplitter + embed_model, docstore)
        │           ├─ pipeline.run(documents, show_progress=True)
        │           │   └─ 哈希去重 + 切块 + 向量化
        │           ├─ VectorStoreIndex(nodes, storage_context)
        │           └─ index.storage_context.persist(kb_storage)
        │               └─（已被补丁替换为流式写盘，见 §4.1）
        │
        ├─[8] _select_retrieval_mode() → 向量 / 聚合 / 全文
        │
        └─[9] 进入 REPL 循环
```

### 5.2 REPL 命令分发调用链

```
while True:
    question = input()
    │
    ├─ exit/quit/退出        → break
    ├─ clear/cls             → os.system("cls")
    │
    ├─ kbs/知识库/kb         → _load_kb_configs() → 显示所有知识库
    │
    ├─ mode                  → _select_retrieval_mode() 切换检索模式
    │
    ├─ rebuild               → shutil.rmtree() → _load_or_build_kb() 重建
    │
    ├─ dedup_status          → 读取 docstore → 统计哈希/文档/向量数
    │
    ├─ embed <路径>          → SimpleDirectoryReader → SentenceSplitter
    │                          → embed_model.get_text_embedding_batch()
    │
    ├─ ingest <路径>         → 分批处理（每批 batch_size=1000 个文件，来自配置）
    │                          ├─ SimpleDirectoryReader → make_pipeline()
    │                          ├─ pipeline.run() → index.insert_nodes()
    │                          ├─ 每批结束落盘检查点（中断可续传，去重自动跳过）
    │                          └─ _input_with_timeout() 批次间确认，超时自动继续
    │
    ├─ list/列表/ls          → 遍历 docstore.docs → 提取 file_path
    │
    ├─[当前模式 = 全文搜索]
    │   └─ _handle_fulltext_query()
    │        ├─ _get_fulltext_searcher()（bigram 分词 + BM25，top_k=20）
    │        └─ _highlight_keywords() → 输出（不调 LLM）
    │
    ├─[聚合查询检测]
    │   └─ _is_aggregate_query(question)?
    │       ├─ Yes → _handle_aggregate_query(question, index)
    │       │        ├─ _extract_keyword_from_query()（策略3时）
    │       │        ├─ 遍历 docstore.docs（tqdm 进度条）
    │       │        └─ 正则匹配 → print_paged() 输出
    │       └─ No  → 走正常查询
    │
    └─[正常查询（向量模式）]
        ├─ query_engine.query(question)
        │   └─ _make_query_engine() 创建：similarity_top_k=10（配置）+ tree_summarize
        ├─ print_paged(str(response))
        │
        └─ 打印检索片段
            ├─ response.source_nodes
            └─ _highlight_relevant_sentences(node.text, question)
                ├─ 正则分句
                ├─ embed_model.get_text_embedding_batch()（句子向量化）
                ├─ embed_model.get_query_embedding()（查询向量化）
                ├─ numpy 计算余弦相似度
                ├─ 选 top-N 句（相似度 >= 0.5）
                └─ _build_ansi_code(cfg) → ANSI 高亮
```

### 5.3 关键数据流

```
[数据源 .md/.txt 文件]
    │
    ▼ SimpleDirectoryReader.load_data()
[Document 对象列表]
    │
    ▼ IngestionPipeline.run()
    │   ├─ SentenceSplitter 切块（512 字符 + 50 重叠）
    │   ├─ SHA256 哈希去重（DUPLICATES_ONLY）
    │   └─ embed_model 向量化
[Node 对象列表（含 embedding）]
    │
    ▼ VectorStoreIndex(nodes, storage_context)
[向量索引]
    │
    ▼ storage_context.persist()（流式写盘补丁已生效）
[storage/<kb_id>/*.json]
    │
    ▼ 下次启动：load_index_from_storage()
[索引对象] → query_engine.query() → 回答
```

***

## 6. 配置文件格式

### 6.1 知识库配置（kb\_configs/\*.yaml）

```yaml
# 必填字段
name: "北京日报"                    # 显示名
data_dir: "D:\\wiki\\beijing_daily\\2026-06-30"  # 数据源目录

# 可选字段
description: "北京日报全文数据"      # 描述
file_exts:                          # 读取的文件扩展名（默认 .md）
  - ".md"
# storage_dir 自动生成为 ./storage/<kb_id>
```

### 6.2 检索参数配置（retrieval\_config.yaml）

集中管理检索参数，CLI（`run_vector_demo.py`）与 Web（`app.py`）共用。
文件不存在时全部使用代码内默认值，程序可独立运行。修改后重启程序生效。

```yaml
vector:
  similarity_top_k: 10          # 向量检索返回片段数（代码兜底默认 5）
  response_mode: tree_summarize # LLM 回答模式

fulltext:
  top_k: 20                     # 全文搜索返回片段数
  bm25_k1: 1.5                  # BM25 词频饱和参数
  bm25_b: 0.75                  # BM25 文档长度归一化参数

chunk:
  chunk_size: 512               # 文本分块大小
  chunk_overlap: 50             # 分块重叠

highlight:
  top_n: 2                      # 每个片段高亮几句
  threshold: 0.5                # 相似度低于此值的句子不高亮

ingest:
  batch_size: 1000              # 单批 ingest 最大文件数，每批结束落盘检查点
  auto_continue_timeout: 10     # 批次间等待确认秒数，超时自动继续
```

### 6.3 高亮配置（highlight\_config.json）

```json
{
    "foreground": "bright_yellow",   // 前景色（可选颜色见 _ANSI_COLOR_CODES）
    "background": null,              // 背景色
    "bold": false,                   // 加粗
    "underline": true,               // 下划线
    "fallback_prefix": ">>>",        // 非 tty 环境的文本标记
    "fallback_suffix": "<<<"
}
```

***

## 7. 新人快速上手

### 7.1 首次运行

```powershell
# 1. 设置 API Key（任选一种 LLM）
$env:DEEPSEEK_API_KEY="你的key"

# 2. 运行 CLI
.venv\Scripts\python.exe run_vector_demo.py

# 或运行 Web UI
.venv\Scripts\python.exe -m streamlit run app.py

# 3. 按提示选择：知识库 → LLM → embedding 模型 → 检索模式
```

详细使用说明见 `MANUAL.md`。

### 7.2 阅读源码顺序

1. **先读** **`main()`**（第 1513 行）—— 理解整体流程
2. **再读 REPL 循环**（`main()` 内部）—— 理解所有命令
3. **按需深入**：
   - 想理解索引构建 → 读 `_load_or_build_kb()` + `make_pipeline()`
   - 想理解查询 → 读 REPL 的"普通提问"分支 + `_highlight_relevant_sentences()`
   - 想理解聚合查询 → 读 `_handle_aggregate_query()`
   - 想理解全文检索 → 读 `_get_fulltext_searcher()` + `_handle_fulltext_query()`
   - 想理解多知识库 → 读 `_select_single_kb()` + `_create_new_kb()`
   - 想理解检索参数 → 读 `_load_retrieval_config()` + `_make_query_engine()`

### 7.3 常见改动位置

| 需求              | 修改位置                                                          |
| --------------- | ------------------------------------------------------------- |
| 调整检索参数（top\_k 等） | `retrieval_config.yaml`（推荐）或 `_load_retrieval_config()` 的默认值   |
| 调整 chunk 大小     | `retrieval_config.yaml` 的 `chunk` 分区                          |
| 新增 LLM 提供商      | `_select_llm_provider()`（141 行）+ `_build_llm()`（170 行）         |
| 新增 embedding 模型 | `_scan_local_embed_models()`（247 行）+ `_build_embed_model()`（327 行） |
| 修改高亮样式          | `highlight_config.json`                                        |
| 新增 REPL 命令      | REPL 循环中添加 `if question.lower() == "xxx":` 分支                 |
| 切换检索模式          | REPL 内输入 `mode`                                                |

***

## 8. 已知限制

1. **聚合查询噪音**：`_handle_aggregate_query()` 的"项目名称"策略用包含"项目"的行匹配，会混入正文段落。精确提取需 LLM 二次过滤或结构化 metadata。
2. **向量检索无 rerank**：`similarity_top_k` 配置为 10（无重排模型的实践取值 5~10）。盲目调大（如 50）会让 `tree_summarize` 每轮塞入约 2.5 万字符，LLM 调用变慢变贵；如需更高召回，应加 reranker（如 bge-reranker）而非堆 top\_k。列举类查询请用聚合模式直遍 docstore。
3. **加载索引慢**：5 万+ chunk 的索引加载约需 190 秒。已加预估进度条（基于存储大小），但无法加速 LlamaIndex 的原子加载操作。
4. **单文件架构**：所有业务逻辑在 `run_vector_demo.py`（1962 行），`app.py` 通过 import 复用。如需进一步扩展，可考虑拆分为 `kb_manager.py` / `query_handler.py` / `highlight.py` 等模块。
5. **持久化已打猴子补丁**：`_streaming_kvstore_persist` 修改了 `SimpleKVStore.persist` 全局行为（import 即生效，含 `app.py`）。升级 llama-index-core 时需确认补丁仍然适用。
