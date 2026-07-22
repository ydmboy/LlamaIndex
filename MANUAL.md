# LlamaIndex RAG 问答系统 — 使用手册

基于 LlamaIndex 的交互式 RAG（检索增强生成）问答系统。
本地 embedding + DeepSeek LLM + IngestionPipeline 哈希去重，支持 Obsidian 笔记/任意文档目录的语义检索与问答。

***

## 一、系统简介

### 架构

支持**多知识库独立管理**：每个知识库有自己的数据源和存储目录，启动前选定一个进入，会话期间锁定，避免内容污染。

```
启动 → 选知识库（kb_configs/*.yaml）→ 选 LLM → 选 embedding → 加载索引
       ↓                                  ↓
       每个库独立：                         会话期间锁定一个库
       - data_dir（数据源）                  所有 query/ingest/rebuild 只作用于此库
       - storage/<kb_id>/（索引）

单个知识库内部数据流：
文档(.md/.txt)
  ↓ SimpleDirectoryReader 读取（filename_as_id=True）
  ↓ IngestionPipeline（哈希去重 DUPLICATES_ONLY）
  ↓   ├─ 计算文档 SHA256 内容哈希
  ↓   ├─ 哈希已存在 → 跳过（不切块、不向量化）
  ↓   └─ 哈希不存在 → SentenceSplitter 切块 → HuggingFaceEmbedding 向量化
  ↓ SimpleVectorStore + SimpleDocumentStore 持久化（./storage/<kb_id>）
查询时（三种检索模式，运行中可用 `mode` 命令切换）：
  模式 1 - 向量检索（默认）：
    问题 → 同模型向量化 → 余弦相似度检索 top-k → DeepSeek 生成回答
  模式 2 - 聚合直遍：
    问题 → 匹配列举类模式（"列出所有X"）→ 直遍 docstore 正则提取 → 100% 覆盖
  模式 3 - 全文搜索：
    问题 → 中文 bigram 分词 → BM25 排序 → 关键词高亮（无 LLM 调用）
```

### 核心组件

| 组件        | 作用              | 当前配置                                                      |
| --------- | --------------- | --------------------------------------------------------- |
| LLM       | 生成最终回答          | DeepSeek `deepseek-chat`，temperature=0                    |
| Embedding | 把文本转成向量         | 启动时交互选择（bge-m3 默认 / gte-Qwen2-7B / bge-small-zh / MiniLM） |
| 切块器       | 把长文档切成小块        | `SentenceSplitter` chunk\_size=512, overlap=50            |
| 向量库       | 存向量并检索          | `SimpleVectorStore`（内存+JSON 持久化，每库独立）                     |
| 读取器       | 读文件成 Document   | `SimpleDirectoryReader`（filename\_as\_id=True）            |
| 去重管道      | 哈希去重防止重复 ingest | `IngestionPipeline` + `DocstoreStrategy.DUPLICATES_ONLY`  |
| 重点句高亮     | 片段内自动划重点，兼顾上下文  | 本地 embedding 余弦相似度 + `highlight_config.json` 样式（纯展示层）     |
| 文档存储      | 存文档内容和哈希映射      | `SimpleDocumentStore`（docstore.json 持久化，每库独立）             |
| 多知识库管理    | 维护多套独立 RAG 系统   | `kb_configs/*.yaml` 配置 + 启动前单选 + `storage/<kb_id>/` 隔离    |

***

## 二、环境准备

### 系统要求

- Windows 10/11（PowerShell 5+）
- Python 3.11（兼容性最佳，3.10 也可）
- NVIDIA GPU（可选，无 GPU 用 CPU 版 torch 也能跑，只是慢）
- 联网（首次部署需下载依赖约 2.5GB）

### 获取 DeepSeek API Key

1. 访问 <https://platform.deepseek.com/api_keys>
2. 创建 API Key，复制保存（形如 `sk-xxxxxxxx`）
3. **安全提醒**：Key 等同于密码，切勿写入文件或提交到 git，只用环境变量传递

***

## 三、首次部署

### 方式 A：从部署包安装（推荐）

适用于拿到 `LlamaIndex_RAG_deploy.rar` 的场景。

```powershell
# 1. 解压到任意目录，例如 D:\LlamaIndexRAG
# 2. PowerShell 进入该目录
cd C:\code\LlamaIndex>

# 3. 执行一键部署脚本
powershell -ExecutionPolicy Bypass -File deploy.ps1
```

`deploy.ps1` 会自动完成：创建 .venv → 安装 uv → 装 torch GPU 版 → 装其余依赖 → 验证 CUDA。

### 方式 B：手动安装

```powershell
# 1. 创建虚拟环境
uv venv .venv --python 3.11

# 2. 装 torch（有 NVIDIA 显卡用 GPU 版，否则用 CPU 版）
# GPU 版（cu121）：
uv pip install torch==2.5.1 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
# CPU 版：
uv pip install torch==2.5.1

# 3. 装其余依赖
uv pip install llama-index-core==0.14.23 `
    llama-index-embeddings-huggingface==0.7.0 `
    llama-index-llms-deepseek==0.3.0 `
    llama-index-readers-file==0.6.0 `
    sentence-transformers==5.6.0 `
    transformers==4.46.3 `
    tqdm
```

### 验证部署

```powershell
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

输出 `CUDA: True` 表示 GPU 可用；`False` 表示用 CPU（功能正常，只是慢）。

***

## 四、运行系统

### 启动

每次启动前先设置环境变量（关掉终端就失效，安全）：

```powershell
# 必需：根据你选的 LLM 提供商设置对应 Key（启动时菜单也会提示）
# DeepSeek（默认）
$env:DEEPSEEK_API_KEY="sk-你的key"
# 或通义千问
# $env:DASHSCOPE_API_KEY="sk-你的key"
# 或智谱 GLM
# $env:ZHIPU_API_KEY="你的key"

# 可选：直接预设 LLM 提供商，跳过启动菜单
# $env:LLM_PROVIDER="deepseek"   # 可选值：deepseek / qwen / zhipu / ollama / custom

# 可选：自定义数据源目录（默认 D:\wiki\beijing_daily\2026-06-30）
$env:DATA_DIR="D:\wiki"

# 可选：自定义索引存储目录（默认 ./storage）
$env:STORAGE_DIR="./storage"

# 可选：预设 embedding 模型，跳过启动菜单
# 不设则启动时自动扫描本地模型并显示菜单（bge-m3 / gte-Qwen2-7B / bge-small-zh / MiniLM 等）
# $env:EMBED_MODEL="C:\code\LlamaIndex\models\bge-m3"   # 推荐：中文好+速度快
# $env:EMBED_MODEL="C:\code\LlamaIndex\models\gte-Qwen2-7B-instruct"   # 最高质量但慢

# 可选：限制首次构建处理的文档数量（调试用，0 表示不限制）
# $env:MAX_DOCS="10"   # 只处理前 10 个文档做快速测试

# 启动
.venv\Scripts\python.exe run_vector_demo.py
```

### 启动流程（依次交互，一直回车即用默认值）

启动后的交互顺序**固定为**：

1. **选择数据集**（第一件事，会话期间锁定，切换需重启）
   - 始终显示菜单（即使只有一个配置）
   - 最后一项 `+ 新建数据集` 可交互式创建新数据集
   - 无配置时直接进入新建流程
2. **选择 LLM 提供商**（默认 DeepSeek）
3. **选择 embedding 模型**（默认 bge-m3）
4. 加载/构建索引 → 进入问答循环

**一直按回车** = 默认数据集 + DeepSeek + bge-m3，最简启动。

### 知识库选择（启动前单选，会话期间锁定）

**启动程序后第一件事就是选择数据集（知识库）**。不同数据集彼此独立，各有独立的数据源和存储目录，避免内容污染。选择后整个会话期间锁定该数据集，所有 ingest/rebuild/query 操作都只作用于这个库，切换需要退出重启。

启动后**始终显示选择菜单**（即使只有一个配置也会显示，让用户明确当前在哪个库），并支持在菜单中**新建数据集**：

```
============================================================
 请选择要进入的数据集（不同数据集彼此独立，避免内容污染）
============================================================
  1. 北京日报 (默认)  [beijing_daily] [已构建]
     数据源: ✓ D:\wiki\beijing_daily\2026-06-30
     存储到: ./storage/beijing_daily
  2. 中国消费者报  [china_consumer_news] [未构建]
     数据源: ✓ D:\wiki\china_consumer_news\2026-06-30
     存储到: ./storage/china_consumer_news
  3. 中国教育报  [china_education_daily] [未构建]
  ...
  36. 浙江日报  [zhejiang_daily] [未构建]
  37. + 新建数据集
============================================================
输入序号 (1-37)，默认 1:
```

当前已配置 **36 个知识库**（4 份中央级 + 25 份省级日报 + 7 份专业类报纸），可用 `generate_kb_configs.py` 批量生成配置。

- **一直回车 = 默认数据集**（`DEFAULT_KB_ID = beijing_daily`）
- 菜单会标注：数据源是否存在（✓/✗）、索引是否已构建
- 选最后一项 `+ 新建数据集` 可交互式创建新数据集（见下文）

#### 新建数据集（菜单中选 `+ 新建数据集`）

在启动菜单中选择"新建数据集"后，按提示输入：

```
------------------------------------------------------------
 新建数据集
------------------------------------------------------------
数据集 ID（英文/数字/下划线，如 literature）: literature
显示名（默认 literature）: 文学作品
数据源目录路径（如 D:\wiki\literature）: D:\wiki\literature
  目录不存在: D:\wiki\literature，仍要创建此数据集吗？(y/N): y
读取的文件扩展名（逗号分隔，默认 .md）: .md,.txt

[已保存] kb_configs\literature.yaml
[新建数据集] 文学作品 [literature]
  数据源: D:\wiki\literature
  存储到: ./storage/literature
  扩展名: ['.md', '.txt']
```

新建后会自动：

1. 保存为 `kb_configs/<kb_id>.yaml`（下次启动可直接在菜单选择）
2. 自动进入该数据集（storage\_dir = `./storage/<kb_id>`）
3. 后续 `ingest`/`rebuild`/`query` 都基于此 storage\_dir

**注意**：新建数据集时数据源目录可以不存在（会提示确认），但构建索引时会报错。请确保数据源目录有文档后再 `rebuild`。

#### 为什么启动前选，不在运行时切换？

| 方式             | 优点                   | 缺点            |
| -------------- | -------------------- | ------------- |
| **启动前选（当前方案）** | 数据集彼此独立，无内容污染；会话状态清晰 | 切换需重启         |
| 运行时热切换         | 切换方便                 | 容易误查错库；索引状态混乱 |

启动前选定的方式更安全：你永远知道当前在哪个数据集里，所有 ingest/rebuild/query 操作都只影响这个库的 `storage/<kb_id>/` 目录。

#### 知识库配置文件

每个数据集对应 `kb_configs/<kb_id>.yaml`，格式：

```yaml
name: "北京日报"                      # 显示名
description: "北京日报全文数据"          # 描述
data_dir: "D:\\wiki\\beijing_daily"   # 数据源目录
file_exts: [".md", ".txt"]            # 读取的文件扩展名
# storage_dir 留空则自动用 ./storage/<kb_id>
```

新增数据集有两种方式：

1. **菜单中新建**（推荐）：启动时选 `+ 新建数据集`，交互式输入，自动保存 yaml
2. **手动创建**：复制 `kb_configs/beijing_daily.yaml.example` 改名修改

`*.yaml.example` 不会被加载，只有 `*.yaml` 会生效。

#### 每个数据集独立存储

```
storage/
├── beijing_daily/    # 北京日报的索引
│   ├── docstore.json
│   ├── index_store.json
│   └── vector_store.json
├── literature/       # 文学作品的索引
└── history/          # 历史资料的索引
```

**ingest 的目标 = 当前选定的数据集的 storage\_dir**。例如选了 `literature`，则 `ingest <文件>` 只会写入 `./storage/literature/`，不会污染其他数据集。

切换 embedding 模型后，需要进入对应数据集执行 `rebuild`。

### LLM 提供商选择

启动后若未设置 `LLM_PROVIDER` 环境变量，会显示菜单：

```
请选择 LLM 提供商：
  1. DeepSeek (deepseek-chat)
  2. 通义千问 (qwen-plus / qwen-turbo)
  3. 智谱 GLM (glm-4 / glm-4-flash)
  4. Ollama 本地模型 (qwen2.5:7b 等)
  5. 自定义 OpenAI 兼容接口
输入序号 (1-5)，默认 1:
```

按数字选择即可，默认 1（DeepSeek）。选择后程序会读取对应的 API Key 环境变量并初始化 LLM。

### Embedding 模型选择

启动后若未设置 `EMBED_MODEL` 环境变量，会自动扫描本地已下载的模型并显示菜单：

```
请选择 embedding 模型：
  0. 手动输入模型名/路径（如 BAAI/bge-m3 或 D:\path\to\model）
  1. all-MiniLM-L6-v2 (本地路径)
  2. bge-m3 (本地路径) (默认)
  3. bge-small-zh-v1.5 (本地路径)
  4. gte-Qwen2-7B-instruct (本地路径)
  5. sentence-transformers/all-MiniLM-L6-v2 (HF缓存)
输入序号 (0-5)，默认 2:
```

**默认值 = bge-m3（中文好 + 速度快）**，一直回车即可使用默认值。

扫描位置：

- `C:\code\LlamaIndex\models\` —— 项目内本地模型（必须含 `config.json`）
- `~/.cache/huggingface/hub/` —— HuggingFace 缓存（必须含 `snapshots/`）

选 0 可手动输入任意模型名（如 `BAAI/bge-m3`，首次会从 HuggingFace 下载）或本地路径。

**自动适配配置**：

- 检测到 Qwen2 系列（路径含 `qwen2` 或 `gte-Qwen`）→ 用专用配置：`trust_remote_code`、`query_instruction`、`max_length=2048`、`embed_batch_size=2`、FP16
- 其他小模型（bge、MiniLM 等）→ 用基础配置：`embed_batch_size=64`、FP16

**切换 embedding 模型注意**：不同模型的向量维度不同（如 MiniLM=384，Qwen2-7B=3584），切换后需要 `rebuild` 重建索引。

### 首次启动 vs 后续启动

| 场景              | 行为                                 | 耗时                        |
| --------------- | ---------------------------------- | ------------------------- |
| 首次（无 ./storage） | 读取 DATA\_DIR 全量文档 → 切块 → 向量化 → 持久化 | 取决于文档量（千级约 1-5 分钟，GPU 加速） |
| 后续（有 ./storage） | 直接加载持久化索引                          | 秒级                        |

### 启动后界面

```
============================================================
 LlamaIndex 交互式 RAG 问答 (DeepSeek + 小模型(FP16))
============================================================
[加载知识库] 北京日报 [beijing_daily]
  数据源: D:\wiki\beijing_daily\2026-06-30
  存储到: ./storage/beijing_daily
[加载] 知识库 [北京日报] 从 ./storage/beijing_daily 读取索引 ...
    存储容量: 1.45 GB | 预估加载 73s
    加载进度: [██████████████████████████████] 100% | 6.3s

============================================================
 当前知识库: 北京日报 [beijing_daily]
 存储位置:   ./storage/beijing_daily (1.45 GB)
 数据源:     D:\wiki\beijing_daily\2026-06-30
 去重: IngestionPipeline 哈希去重 (DUPLICATES_ONLY)
 检索模式:   向量检索 [vector]
============================================================
索引就绪。输入问题即可获取回答。
命令：  exit / quit / 退出  -> 结束；  clear  -> 清屏
      kbs  -> 查看所有可用知识库（切换需重启程序）
      mode -> 切换检索模式（向量/聚合/全文）
      rebuild  -> 重建当前知识库索引
      embed <路径>  -> 向量化单个文件并打印结果（不写入主索引）
      ingest <路径>  -> 把指定文件/文件夹增量加入当前知识库（自动去重）
      list  -> 列出当前知识库中所有已 ingest 的文件路径
      dedup_status  -> 查看去重统计（哈希数/文档数/跳过数）
      （长输出超过 30 行自动分页，按空格翻页，q 退出）
------------------------------------------------------------

[向量] 你的问题>
```

提示符前缀 `[向量]` / `[聚合]` / `[全文]` 表示当前检索模式，用 `mode` 命令切换。

### 命令一览

所有命令都作用于**当前知识库**（启动时选定的那个）。要切换知识库请退出重启。

| 命令                     | 说明                           |
| ---------------------- | ---------------------------- |
| `kbs`                  | 查看所有可用知识库（标注 `[当前]`），切换需重启程序 |
| `mode`                 | 切换检索模式（向量检索 / 聚合直遍 / 全文搜索）   |
| `rebuild`              | 重建当前知识库索引（删除后重新读取数据源）        |
| `list`                 | 列出当前知识库中已 ingest 的文件路径       |
| `dedup_status`         | 查看当前知识库的去重统计                 |
| `embed <路径>`           | 向量化单个文件并打印结果（不写入索引）          |
| `ingest <路径>`          | 增量加入当前知识库（自动去重）              |
| `clear`                | 清屏                           |
| `exit` / `quit` / `退出` | 退出（重启后可选其他知识库）               |

### 首次构建的去重输出

首次构建或 ingest 时会显示去重统计和存储容量：

```
[构建] 读取数据源: C:\Users\opq\Documents\Obsidian Vault
      只读取 .md 文件（自动跳过 .obsidian 配置目录等）
    -> 发现 1058 个 .md 文件，开始逐个读取...
  读取进度: [████████████████████████████████░░] 1000/1058 (94.5%) | 45.2file/s | ETA 1s
  ▶ notes/python/async.md
    notes/go/goroutine.md
    notes/rust/ownership.md
    notes/算法/动态规划.md
    notes/数据库/索引优化.md
  [OK] 读取完成: 1058 个文件，1058 个文档 | 耗时 23.4s
[构建] 运行 IngestionPipeline（哈希去重 + 切块 + 向量化）...
Applying transformations:  50%|██████████| 1/2 [00:09<00:09,  9.30s/it]   ← 第 1 步：切块
Applying transformations: 100%|██████████| 2/2 [00:30<00:39, 19.50s/it]   ← 第 2 步：向量化
    -> 索引已持久化到 ./storage，下次启动将直接加载
    -> 去重统计: 输入 1058 文档 → 新增 1058 → 跳过 0 个重复
    -> docstore 共跟踪 1058 个唯一哈希
    -> 存储容量: 12.4 GB
```

**文档读取阶段进度显示**（ANSI Live 显示，仅真实终端可见）：

- **顶部**：固定进度条 `█░` + 文件计数 + 百分比 + 速度 + ETA
- **底部**：最近 5 个文件的相对路径，最新一个用 `▶` 标记
- **完成行**：`[OK]` 汇总总文件数、文档数、耗时
- **IDE 内置终端 / 管道重定向**：自动降级为每 50 个文件打印一行进度（避免 ANSI 控制码污染日志）

**IngestionPipeline 进度条（`Applying transformations`）说明**：

该进度条统计的是两个 transformation 步骤，**不是文档数**：

| 步骤    | 进度      | 做什么                               | 耗时               |
| ----- | ------- | --------------------------------- | ---------------- |
| 第 1 步 | 50% 完成  | `SentenceSplitter` 切块（CPU 操作）     | 快（\~9 秒/千文档）     |
| 第 2 步 | 100% 完成 | `embed_model` 对每个 chunk 做 7B 模型推理 | **主要耗时**（GPU 操作） |

第 2 步（embedding）是整个构建过程的瓶颈，LlamaIndex 在此阶段不提供 chunk 级进度条，看起来会"卡在 50%"，但 GPU 实际在全力运行。可通过另一终端运行 `nvidia-smi -l 3` 监控 GPU 利用率确认是否在工作。

### Web UI 模式（Streamlit）

除 CLI 交互式问答外，项目还提供浏览器 Web 界面（`app.py`），复用 `run_vector_demo.py` 的全部函数，适合展示和远程操作。

#### 启动

```powershell
$env:STREAMLIT_HOME="c:\code\LlamaIndex\.streamlit"
.venv\Scripts\streamlit.exe run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

访问 http://localhost:8501

#### 功能

| 区域 | 功能 |
|------|------|
| **侧边栏** | 知识库选择（36 个）+ 检索模式切换（向量/聚合/全文）+ 模型状态 |
| **💬 问答 Tab** | 聊天界面，根据当前检索模式分派（向量=LLM回答 / 聚合=直遍 / 全文=BM25），支持来源片段展开 |
| **📥 Ingest Tab** | 目录批量导入（进度条 4 阶段）+ 文件上传 + 操作日志 |
| **🗂️ 知识库管理 Tab** | 36 个知识库列表，显示索引状态/大小/节点数，支持加载/删除索引 |

#### 加载知识库进度条

点击「加载/切换知识库」后显示预估进度条（基于存储大小 × 20 MB/s 速率），后台线程每 200ms 更新，实际完成后跳到 100%。

#### 注意事项

- Web UI 只能加载**已构建索引**的知识库（`storage/<kb_id>/docstore.json` 存在）
- 未构建索引的知识库会显示"⬜ 未构建索引"提示，需先在 CLI 中运行 `run_vector_demo.py` 构建索引
- `--browser.gatherUsageStats false` 必须加（避免 Streamlit 写入无权限目录）

***

## 五、REPL 命令详解

在 `[模式] 你的问题>` 提示符后输入命令，回车执行。提示符前缀 `[向量]` / `[聚合]` / `[全文]` 表示当前检索模式。

### 5.0 mode — 切换检索模式

系统支持三种检索模式，运行中随时切换，无需重启：

```
[向量] 你的问题> mode

请选择检索模式：
  1. 向量检索（语义匹配，LLM 生成回答） (当前)
  2. 聚合直遍（正则匹配，100%覆盖，适合列举类查询）
  3. 全文搜索（倒排索引 + BM25 排序，关键词精确匹配）
输入序号 (1-3)，默认 1: 2

[聚合] 你的问题>
```

三种模式对比：

| 模式 | 原理 | 覆盖率 | 语义理解 | 耗时 | 适用场景 |
|------|------|--------|---------|------|---------|
| **向量检索**（默认） | 查询和文档都转向量，算余弦相似度，返回 top-k | 低（top_k=5） | ✅ 有 | 快（1-2s） | 语义问答，LLM 生成回答 |
| **聚合直遍** | 遍历 docstore 所有节点，用正则提取匹配行 | 100% | ❌ 无 | 中（2-5s） | 列举类查询（"列出所有文章标题""有哪些项目"） |
| **全文搜索** | 中文 bigram 分词 + BM25 排序 + 关键词高亮 | 高 | 部分 | 首次构建 13s + 查询 11s | 关键词精确匹配，无 LLM 调用 |

**使用建议**：
- 语义问答用**向量检索**（默认）
- "列出所有 X""有哪些 X"等聚合查询用**聚合直遍**
- 精确关键词搜索用**全文搜索**

**全文搜索首次使用**：BM25 倒排索引懒加载，首次查询时构建（约 13 秒），后续查询直接复用缓存。

### 5.1 提问（核心功能）

直接输入任意自然语言问题：

```
你的问题> 2026年6月30日的人民日报头版报道了什么？
```

输出格式：

```
回答:
<DeepSeek 生成的答案>

============================================================
检索到 2 个相关片段:
============================================================

--- 片段 1 | 来源: D:\wiki\people_daily\2026-06-30\人民日报-2026-06-30-全刊.md ---
<检索到的原文片段，其中与问题最相关的 1-2 句会被高亮>

--- 片段 2 | 来源: D:\wiki\jiefang_daily\2026-06-30\解放日报-2026-06-30-全刊.md ---
<检索到的原文片段，其中与问题最相关的 1-2 句会被高亮>

============================================================
```

**两层输出**：先给答案，再给原文片段，方便核对 LLM 是否在"瞎编"。

#### 重点句高亮（自动）

每个检索片段会**自动高亮与问题最相关的 1-2 句话**，兼顾上下文和重点：

- **划重点执行者**：本地 embedding 模型（已加载在内存，零 API 调用）
- **工作原理**：把片段按中文标点（`。！？\n`）切句 → 批量计算每句与查询的余弦相似度 → 选 top-N 句（相似度需 ≥ 0.5）标记为重点
- **显示方式**：
  - **独立 PowerShell / Windows Terminal**（tty 环境）：根据 `highlight_config.json` 配置生成 ANSI 样式（默认亮黄色文字 + 下划线）
  - **IDE 内置终端 / 管道重定向**（非 tty）：用 `fallback_prefix`/`fallback_suffix` 标记包裹（默认 `>>>重点句<<<`），避免 ANSI 转义码污染输出
- **保留上下文**：非重点句正常显示，重点句在原文位置高亮，不破坏阅读顺序
- **可调参数**（在 `run_vector_demo.py` 顶部配置区）：
  - `HIGHLIGHT_TOP_N = 2`：每个片段高亮几句
  - `HIGHLIGHT_THRESHOLD = 0.5`：相似度阈值，低于此值不高亮（避免噪音）
  - `HIGHLIGHT_SENTENCE_SPLIT = r"[。！？\n]+"`：中文分句正则
  - `HIGHLIGHT_CONFIG_FILE = "highlight_config.json"`：高亮样式配置文件路径

**样式自定义**：编辑项目根目录下的 `highlight_config.json` 文件即可调整高亮样式（颜色、下划线、斜体、加粗等），修改后重启程序生效。详见 [§ 六 - 展示层样式配置](#展示层样式配置highlight-config-json)。

**注意**：高亮是纯展示层功能，不改变索引、不改变检索逻辑、不需要 rebuild。如果 embedding 模型调用失败，会自动降级为不高亮，不影响主流程。

### 5.2 ingest — 增量加入文档（自动去重）

把指定文件或文件夹加入主索引，**自动跳过内容哈希已存在的重复文档**，加入后即可被问答检索。

```
你的问题> ingest D:\wiki
你的问题> ingest D:\wiki\people_daily\2026-06-30\人民日报-2026-06-30-全刊.md
你的问题> ingest "D:\带空格的目录\笔记.md"
你的问题> 加入 D:\notes.txt          # 中文别名
```

| 参数    | 说明                   |
| ----- | -------------------- |
| 单文件   | 加入该文件（.md/.txt 等）    |
| 文件夹   | 递归读取该文件夹下所有 .md/.txt |
| 路径含空格 | 用双引号或单引号包起来          |

执行过程会显示进度条、去重统计和存储容量变化：

```
[ingest] 目标: D:\wiki
    -> 加载 36 个文档，开始增量写入（哈希去重）...
ingest:   0%|          | 0/36 [00:00<?, ?doc/s]
    [1/36] 处理: 人民日报-2026-06-30-全刊.md
ingest:   3%|▏         | 1/36 [00:03<01:48, 3.20s/doc]
...
ingest: 100%|██████████| 36/36 [01:52<00:00, 3.11s/doc]
    -> 完成！新增 28 个文档，跳过 8 个重复
    -> 已持久化到 ./storage，docstore 共跟踪 28 个唯一哈希
    -> 存储容量: 335.21 MB（本次 +15.72 MB）
```

**特点**：

- **自动去重**：基于 SHA256 内容哈希，完全相同的文档不会重复 ingest
- 增量追加，不影响已有索引内容
- 自动持久化到 `./storage`，退出后下次启动仍在
- 复用已加载的 embedding 模型，无需重新初始化
- 去重不依赖文档格式，对 .md/.txt/.pdf 等任何格式通用

### 5.3 embed — 仅向量化查看

向量化单个文件并打印结果，**不写入主索引**。用于调试/理解向量化过程。

```
你的问题> embed D:\wiki\people_daily\2026-06-30\人民日报-2026-06-30-全刊.md
```

输出：

```
[向量化] 读取文件: D:\wiki\...\人民日报-2026-06-30-全刊.md
    -> 加载 1 个文档
    -> 切成 24 个 chunk，开始向量化（本地 embedding）...

============================================================
向量化完成！共 24 个向量，维度 = 384
============================================================
  chunk  1 | 前3维=[0.0123, -0.034, 0.056] | 内容预览: 今日人民日报头版...
  chunk  2 | 前3维=[-0.021, 0.048, 0.019] | 内容预览: 本报讯 记者从...
  ...
============================================================
```

**用途**：查看某个文档被切成几块、每块的向量长什么样，不污染主索引。

### 5.4 list — 列出已索引文件

查看主索引里当前有哪些文件：

```
你的问题> list
```

输出：

```
============================================================
主索引中已 ingest 的文件路径：
============================================================
    1. D:\wiki\people_daily\2026-06-30\人民日报-2026-06-30-全刊.md
    2. D:\wiki\jiefang_daily\2026-06-30\解放日报-2026-06-30-全刊.md
    ...
============================================================
共 28 个不同来源文件（28 个文档/节点，856 个向量）
============================================================
```

### 5.5 dedup\_status — 查看去重统计

查看哈希去重的运行状态：

```
你的问题> dedup_status
```

输出：

```
============================================================
去重统计 (IngestionPipeline + DUPLICATES_ONLY)
============================================================
  docstore 跟踪的唯一哈希数: 28
  docstore 存储的文档/节点数: 28
  不同来源文件数:           28
  向量库中的向量数:          856
  存储目录总容量:            319.49 MB
------------------------------------------------------------
说明：
  - 唯一哈希 = 曾 ingest 过的去重后文档数
  - 重复文档在 ingest 时被自动跳过，不会出现在上述计数中
  - 跳过数 = 历次 ingest 时输入文档数 - 新增哈希数（不累计）
============================================================
```

| 指标      | 含义                                   |
| ------- | ------------------------------------ |
| 唯一哈希数   | docstore 中跟踪的 SHA256 哈希总数（= 去重后的文档数） |
| 文档/节点数  | docstore 中存储的文档对象数                   |
| 不同来源文件数 | 去重后保留的来源文件数                          |
| 向量数     | 向量库中实际存储的 chunk 向量数                  |
| 存储目录总容量 | `./storage` 目录占用磁盘大小                 |

### 5.6 rebuild — 全量重建

清空 `./storage` 并从 `DATA_DIR` 重新读取所有文档构建索引（含去重）。

```
你的问题> rebuild
```

输出示例：

```
已删除旧索引（原 319.49 MB），正在重新读取笔记并构建 ...
[构建] 读取数据源: D:\wiki
    ...
    -> 存储容量: 305.27 MB
重建完成。当前存储容量: 305.27 MB
```

**何时使用**：

- 更换了 `DATA_DIR` 指向的目录
- 修改了 `EMBED_MODEL`（不同模型向量维度不同，必须重建）
- 想清掉所有 ingest 的外部文档，回到纯净状态
- 索引损坏/异常
- 想重置去重哈希表（清掉所有历史记录）

**警告**：会删除现有索引和去重记录，重新构建耗时取决于文档量。

### 5.7 长输出分页（自动）

当某条命令的输出超过 **30 行** 时（如查询返回的长答案、多个 source\_nodes 原文、`list` 列出大量文件、`embed` 切出大量 chunk），系统会**自动启用分页**，避免 PowerShell 终端只显示尾部内容。

**实现机制**（`print_paged()` 函数，零新增依赖）：

| 运行环境                                   | 分页方式                            | 说明                                   |
| -------------------------------------- | ------------------------------- | ------------------------------------ |
| 真实 PowerShell / cmd / Windows Terminal | `pydoc.pager` → 系统自带 `more.com` | Python 标准库机制，是 `help()` 函数同款方案，最成熟稳定 |
| IDE 捕获输出 / 管道重定向 / CI 日志               | 直接全打印                           | 非交互环境无法接收按键，强行分页只会刷屏                 |

**操作方式**（真实终端中）：

- 按空格 / Enter → 翻到下一页
- 按 `q` → 退出分页（剩余内容不再显示）
- 上下方向键 → 上下滚动（`more.com` 行为）

**短内容不受影响**：行数 ≤ 30 时直接 `print`，不弹分页器，避免短内容也打扰用户。

**在 IDE 内置终端看到全打印是正常的**：Trae / VS Code 等 IDE 的输出捕获会让 `stdin.isatty()` 返回 False，系统识别为非交互环境自动降级。要看真实分页效果，请在独立的 PowerShell / Windows Terminal 窗口里运行 `python run_vector_demo.py`。

### 5.8 其他命令

| 命令                           | 作用             |
| ---------------------------- | -------------- |
| `clear` 或 `cls`              | 清屏             |
| `exit` / `quit` / `退出` / `q` | 退出程序           |
| `Ctrl+C`                     | 退出（已做异常捕获，不报错） |

***

## 六、配置项详解

所有配置均通过环境变量设置，无需改代码。

#### LLM 提供商选择（启动时交互选择或环境变量预设）

启动时会显示菜单让你选 LLM 提供商，也可通过 `LLM_PROVIDER` 环境变量跳过菜单：

| `LLM_PROVIDER` | 显示名称          | 需要的环境变量                                           | 默认模型                               |
| -------------- | ------------- | ------------------------------------------------- | ---------------------------------- |
| `deepseek`（默认） | DeepSeek      | `DEEPSEEK_API_KEY`                                | `deepseek-chat`                    |
| `qwen`         | 通义千问          | `DASHSCOPE_API_KEY`                               | `qwen-plus`（可用 `QWEN_MODEL` 覆盖）    |
| `zhipu`        | 智谱 GLM        | `ZHIPU_API_KEY`                                   | `glm-4`（可用 `ZHIPU_MODEL` 覆盖）       |
| `ollama`       | Ollama 本地     | 无（需先启动 Ollama 服务）                                 | `qwen2.5:7b`（可用 `OLLAMA_MODEL` 覆盖） |
| `custom`       | 自定义 OpenAI 兼容 | `CUSTOM_API_KEY`、`CUSTOM_API_BASE`、`CUSTOM_MODEL` | `gpt-3.5-turbo`                    |

> 说明：千问和智谱都提供 OpenAI 兼容接口，依赖 `llama-index-llms-openai-like`（已安装）。
> Ollama 依赖 `llama-index-llms-ollama`（已安装），需先在本地启动 Ollama 服务（`ollama serve`）。

#### 通用配置

| 环境变量                | 必需  | 默认值                                | 说明                                                  |
| ------------------- | --- | ---------------------------------- | --------------------------------------------------- |
| `LLM_PROVIDER`      | 否   | 空（启动时交互选择）                         | LLM 提供商：`deepseek`/`qwen`/`zhipu`/`ollama`/`custom` |
| `DEEPSEEK_API_KEY`  | 视选择 | 无                                  | DeepSeek API Key（选 deepseek 时必需）                    |
| `DASHSCOPE_API_KEY` | 视选择 | 无                                  | 阿里云 DashScope API Key（选 qwen 时必需）                   |
| `ZHIPU_API_KEY`     | 视选择 | 无                                  | 智谱 API Key（选 zhipu 时必需）                             |
| `DATA_DIR`          | 否   | `D:\wiki\beijing_daily\2026-06-30` | 数据源目录（无 kb\_configs 时使用）                            |
| `STORAGE_DIR`       | 否   | `./storage`                        | 索引持久化目录（无 kb\_configs 时使用）                          |
| `EMBED_MODEL`       | 否   | 空（启动时交互选择）                         | embedding 模型（本地路径或 HF Hub ID）。不设则启动时扫描本地模型并显示菜单     |
| `MAX_DOCS`          | 否   | `0`（不限制）                           | 调试用：限制首次构建处理的文档数，如 `10` 只处理前 10 个                   |

### 多知识库配置（非环境变量，在 `run_vector_demo.py` 顶部）

| 配置项              | 默认值             | 说明                                  |
| ---------------- | --------------- | ----------------------------------- |
| `KB_CONFIGS_DIR` | `kb_configs`    | 知识库 YAML 配置目录，每个 `.yaml` 文件是一个知识库   |
| `DEFAULT_KB_ID`  | `beijing_daily` | 启动菜单的默认知识库 ID（一直回车即选此库）。找不到则回退第 1 个 |

每个 `kb_configs/<kb_id>.yaml` 文件字段：

| 字段            | 必需 | 说明                                |
| ------------- | -- | --------------------------------- |
| `name`        | 是  | 显示名（如 "北京日报"）                     |
| `description` | 否  | 描述文本，菜单中展示                        |
| `data_dir`    | 是  | 数据源目录路径                           |
| `storage_dir` | 否  | 索引存储目录，留空则自动用 `./storage/<kb_id>` |
| `file_exts`   | 否  | 读取的文件扩展名列表，默认 `[".md"]`           |

**注意**：`*.yaml.example` 不会被加载，只有 `*.yaml` 会生效。新增知识库时复制 `.example` 改名即可。

### 展示层配置（非环境变量，在 `run_vector_demo.py` 顶部）

| 配置项                        | 默认值                     | 说明                         |
| -------------------------- | ----------------------- | -------------------------- |
| `HIGHLIGHT_TOP_N`          | `2`                     | 每个检索片段高亮几句话（按相似度排序取 top-N） |
| `HIGHLIGHT_THRESHOLD`      | `0.5`                   | 相似度阈值，低于此值的句子不高亮（避免噪音）     |
| `HIGHLIGHT_SENTENCE_SPLIT` | `r"[。！？\n]+"`           | 中文分句正则，按句号/感叹号/问号/换行切分     |
| `HIGHLIGHT_CONFIG_FILE`    | `highlight_config.json` | 高亮样式配置文件路径                 |
| `PAGER_THRESHOLD_LINES`    | `30`                    | 输出超过此行数触发分页                |

### 展示层样式配置（highlight\_config.json）

项目根目录下的 `highlight_config.json` 用于配置重点句的高亮样式（颜色、下划线、斜体、加粗等）。修改后重启程序生效，无需 rebuild 索引。

#### 配置项说明

| 字段                | 类型          | 说明                                  |
| ----------------- | ----------- | ----------------------------------- |
| `foreground`      | 字符串或 `null` | 前景色（文字颜色），可选值见下表；设为 `null` 表示不设置前景色 |
| `background`      | 字符串或 `null` | 背景色；设为 `null` 表示不设置背景色              |
| `bold`            | 布尔          | 加粗（`true`/`false`）                  |
| `italic`          | 布尔          | 斜体（部分终端不支持，可能显示为反相）                 |
| `underline`       | 布尔          | 下划线                                 |
| `strikethrough`   | 布尔          | 删除线                                 |
| `reverse`         | 布尔          | 反相显示（前景/背景互换）                       |
| `fallback_prefix` | 字符串         | 非 tty 环境（IDE 捕获/管道）下包裹重点句的前缀        |
| `fallback_suffix` | 字符串         | 非 tty 环境下包裹重点句的后缀                   |

#### 可选颜色名

| 标准色（30-37）                                                     | 亮色（90-97）                                                                                                              |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `black`、`red`、`green`、`yellow`、`blue`、`magenta`、`cyan`、`white` | `bright_black`、`bright_red`、`bright_green`、`bright_yellow`、`bright_blue`、`bright_magenta`、`bright_cyan`、`bright_white` |

无效的颜色名会被忽略（不报错，跳过该项）。

#### 配置示例

```json
{
    "foreground": "bright_yellow",
    "background": null,
    "bold": false,
    "italic": false,
    "underline": true,
    "strikethrough": false,
    "reverse": false,
    "fallback_prefix": ">>>",
    "fallback_suffix": "<<<"
}
```

#### 常见样式组合参考

| 描述           | 配置                                                     | 生成的 ANSI 码 |
| ------------ | ------------------------------------------------------ | ---------- |
| 亮黄+下划线（默认）   | `foreground: bright_yellow, underline: true`           | `4;93`     |
| 白字红底+加粗      | `foreground: white, background: red, bold: true`       | `1;37;41`  |
| 亮青+加粗+斜体+下划线 | `foreground: bright_cyan, bold/italic/underline: true` | `1;3;4;96` |
| 反相显示         | `reverse: true`                                        | `7`        |
| 绿色文字（原始方案）   | `foreground: green`                                    | `32`       |
| 亮红+删除线       | `foreground: bright_red, strikethrough: true`          | `9;91`     |

#### 容错机制

- 配置文件不存在：使用内置默认值（亮黄+下划线）
- 配置文件 JSON 格式错误：打印警告，使用内置默认值
- 颜色名无效：跳过该项，不报错
- embedding 模型调用失败：原样返回文本，不高亮

### 常用 embedding 模型推荐

| 模型                                       | 参数量  | 维度   | 显存      | 适用场景                    | 本地已下载   |
| ---------------------------------------- | ---- | ---- | ------- | ----------------------- | ------- |
| `Alibaba-NLP/gte-Qwen2-7B-instruct`      | 7B   | 3584 | \~15GB  | 中文 SOTA，需要 16GB 显存      | ✅ 29GB  |
| `BAAI/bge-m3`                            | 568M | 1024 | \~3GB   | 中英混合，输入 8192 token，性价比高 | ✅ 2.2GB |
| `BAAI/bge-large-zh-v1.5`                 | 326M | 1024 | \~2GB   | 中文，平衡速度与质量              | ❌       |
| `BAAI/bge-small-zh-v1.5`                 | 24M  | 512  | \~0.3GB | 中文，速度最快，质量中等            | ✅ 92MB  |
| `sentence-transformers/all-MiniLM-L6-v2` | 22M  | 384  | \~0.3GB | 英文，速度快（最大输入 256 token）  | ✅ 87MB  |

**gte-Qwen2-7B-instruct 特殊要求**：

- **必须 FP16 加载**：`model_kwargs={"torch_dtype": torch.float16}`，否则显存翻倍爆显存
- **必须 trust\_remote\_code=True**：模型用了自定义代码
- **必须 query\_instruction**：query 端需加 `Instruct: ...\nQuery: `    前缀，否则检索质量打折
- **必须 pooling="last"**：用 last\_token\_pool，与模型训练方式一致
- **建议 embed\_batch\_size=2\~8**：7B 模型显存压力大，batch 调小防爆显存（默认 8，显存紧张时改 2）
- **建议 max\_length=2048\~8192**：模型支持 32k，但 8192 已够用且省显存（显存紧张可改 2048）
- **首次下载约 29GB**：需联网从 HuggingFace 下载，已预下载到 `C:\code\LlamaIndex\models\gte-Qwen2-7B-instruct\`，默认直接从本地路径加载
- **构建索引时间约 30-60 分钟**（1058 个 .md 文档）：远慢于 bge-small 的 1 分钟
- **运行时不要开浏览器/其他 AI 工具**：显存几乎占满（15/16GB）
- **调试技巧**：可用 `$env:MAX_DOCS="10"` 限制只处理前 10 个文档，快速验证构建流程和进度条显示，约 30-60 秒即可完成

#### **换模型后必须** **`rebuild`**：不同模型向量维度不同，旧索引不兼容。系统启动时会自动校验维度一致性，若发现索引维度与当前模型不一致，会打印警告并拦截 `ingest` 操作，提示执行 `rebuild`。

***

## 七、典型工作流

### 场景 1：首次使用，问答自己的笔记

```powershell
$env:DEEPSEEK_API_KEY="sk-xxx"
$env:DATA_DIR="C:\Users\我\Documents\我的笔记"
.venv\Scripts\python.exe run_vector_demo.py
# 首次启动自动构建索引（等几分钟）
# 构建完成后：
你的问题> 我的笔记里关于项目管理的要点有哪些？
```

### 场景 2：增量加入外部文档

```powershell
你的问题> ingest D:\参考资料\行业报告
# 等待进度条跑完
你的问题> list
# 确认新文件已加入
你的问题> 行业报告里对2026年趋势的预测是什么？
```

### 场景 3：调试某个文档的向量化

```powershell
你的问题> embed "D:\笔记\某篇.md"
# 查看 chunk 数量和向量预览，判断切块是否合理
```

### 场景 4：换中文 embedding 模型提升检索质量

```powershell
# 1. 退出当前 REPL
你的问题> exit

# 2. 设置新模型（默认已是 gte-Qwen2-7B-instruct，下面是切换其他模型的示例）
$env:EMBED_MODEL="BAAI/bge-m3"  # 显存不够时的备选

# 3. 重新启动
.venv\Scripts\python.exe run_vector_demo.py

# 4. 因为模型变了，storage 里的旧索引不兼容，需要 rebuild
你的问题> rebuild
# 等待重建完成

# 5. 重新提问
你的问题> 描写景色的句子有哪些？
```

### 场景 5：维护多个独立知识库（文学 / 历史 / 政治互不污染）

**目标**：同时维护"北京日报""文学""历史""政治"等多个数据集，互不污染，启动时选一个进入。

**步骤**：

1. **创建新数据集**（两种方式）：

   **方式 A：菜单中新建（推荐）** — 启动时选 `+ 新建数据集`：
   ```
   ============================================================
    请选择要进入的数据集（不同数据集彼此独立，避免内容污染）
   ============================================================
     1. 北京日报 (默认)  [beijing_daily] [已构建]
     2. + 新建数据集
   ============================================================
   输入序号 (1-2)，默认 1: 2

   数据集 ID（英文/数字/下划线，如 literature）: literature
   显示名（默认 literature）: 文学作品
   数据源目录路径（如 D:\wiki\literature）: D:\wiki\literature
   读取的文件扩展名（逗号分隔，默认 .md）: .md,.txt

   [已保存] kb_configs\literature.yaml
   [新建数据集] 文学作品 [literature]
     数据源: D:\wiki\literature
     存储到: ./storage/literature
   ```
   **方式 B：手动创建 yaml** — 在 `kb_configs/` 下创建 `.yaml`：
   ```yaml
   # kb_configs/literature.yaml
   name: "文学作品"
   description: "中外文学作品语料"
   data_dir: "D:\\wiki\\literature"
   file_exts: [".md", ".txt"]
   ```
2. 下次启动时菜单会显示所有数据集（包括新建的）：
   ```
   ============================================================
    请选择要进入的数据集（不同数据集彼此独立，避免内容污染）
   ============================================================
     1. 北京日报 (默认)  [beijing_daily] [已构建]
     2. 文学作品  [literature] [未构建]
     3. 历史资料  [history] [未构建]
     4. + 新建数据集
   ============================================================
   输入序号 (1-4)，默认 1:
   ```
3. 选定后整个会话只在这个数据集内操作：
   - `ingest <文件>` → 写入 `./storage/<当前kb_id>/`
   - `rebuild` → 重建 `./storage/<当前kb_id>/`
   - `query` → 只检索 `./storage/<当前kb_id>/` 的向量
4. 查看所有数据集状态：`kbs` 命令显示每个库的激活状态、数据源、存储位置和容量。
5. **切换到其他数据集**：退出程序（`exit`）后重新启动，选择另一个。

**为什么不在运行时切换？** 启动前选定能保证会话状态清晰：你永远知道当前在哪个库，所有操作只作用于这个库的 `storage/<kb_id>/` 目录，不会误查错库或误 ingest 到别的库。索引、哈希表、向量数据都完全隔离。

### 场景 6：部署到另一台机器

参见部署包内的 `README_DEPLOY.txt` 和 `deploy.ps1`。

***

## 八、检索质量与原理

### 数据流七个环节

1. **读取器** `SimpleDirectoryReader`：读文件成 Document，不解析 markdown 结构
2. **去重管道** `IngestionPipeline`：计算文档 SHA256 哈希，重复文档直接跳过
3. **切块器** `SentenceSplitter`：按句子边界 + 长度切块，不识别语义类型
4. **embedding 模型** `gte-Qwen2-7B-instruct`：把文本块转成向量（3584 维），捕捉语义相似度
   - query 端自动加 `Instruct: ...\nQuery: `    前缀，文档端不加（官方推荐）
   - 用 `last_token_pool` 池化策略，FP16 推理
5. **向量库** `SimpleVectorStore`：暴力余弦相似度检索，无阈值过滤
6. **检索+生成**：取 top-k 片段交给 DeepSeek 生成回答
7. **重点句高亮**（展示层）：用本地 embedding 在每个片段内找 top-N 相似句高亮，不改变检索结果

### 哈希去重原理

```
文档 A → SHA256("文档A内容") = hash_a
  ├─ hash_a 不在 docstore → 切块 + 向量化 + 存储哈希
  └─ 记录: docstore[hash_a] = doc_id_a

文档 B（内容与 A 完全相同）→ SHA256("文档B内容") = hash_a
  └─ hash_a 已在 docstore → 跳过（不切块、不向量化、不存储）
```

**去重粒度**：文档级（一个 .md 文件 = 一个文档）。若两个文件内容完全相同，第二个自动跳过。若两个文件只有部分内容相同（如报纸转载同一篇文章），两个文件都会被索引（因为整体哈希不同）。

**去重特点**：

- 基于 SHA256 内容哈希，零误判零漏判
- 完全不依赖文档格式，.md/.txt/.pdf 通用
- 哈希映射持久化在 `docstore.json`，重启后仍记得已 ingest 的内容
- 第二次 ingest 同一批文件，秒级跳过所有重复

### 当前配置的已知局限

| 问题                           | 影响           | 修法                                                       |
| ---------------------------- | ------------ | -------------------------------------------------------- |
| 去重粒度为文档级                     | 部分重叠的文档不会被去重 | 后续可加语义去重（Phase 2）                                        |
| chunk\_size=512 > 模型最大输入 256 | chunk 后半句被截断 | 改 `Settings.chunk_size=256` 或换 bge 模型                    |
| embedding 模型偏英文              | 中文检索召回率低     | 换 `Alibaba-NLP/gte-Qwen2-7B-instruct`（默认）或 `BAAI/bge-m3` |
| top\_k 默认 2                  | 枚举类问题遗漏多     | 改 `similarity_top_k=5`（需改代码）                             |
| 无相似度阈值                       | 不相关结果也返回     | 在 prompt 里让 LLM 判断                                       |

### 不会自动做的事

- **不会标签化**：不会自动识别"景色描写/转折/三字短语"等自定义分类
- **不会分类**：不会给文档打类别标签
- **不会提取结构**：不会解析标题层级、列表、frontmatter
- **不会语义去重**：不会识别"略改的转载"（标题改了几个字但内容几乎相同的文章）

如需标签化，需在 ingest 时用 LLM 预打标签存入 metadata，或用专门的分类模型。

***

## 九、常见问题排查

### Q1：报错 `请先设置环境变量 DEEPSEEK_API_KEY`

```powershell
$env:DEEPSEEK_API_KEY="sk-你的key"
```

每次开新终端都要重新设置（环境变量不持久）。

### Q2：报错 `ImportError: cannot import name 'TransformGetItemToIndex'`

`transformers` 版本与 `torch` 不兼容。降级 transformers：

```powershell
uv pip install "transformers==4.46.3"
```

### Q3：报错 `AttributeError: 'VectorStoreIndex' object has no attribute 'insert_document'`

API 用错了，正确方法是 `index.insert(doc)`，不是 `insert_document`。已修复，若仍出现说明用的是旧版脚本。

### Q3.5：ingest 显示"跳过 N 个重复"但我确定是新文件

哈希去重基于文档**完整内容**，不是文件名。如果两个文件内容完全相同（即使路径不同），第二个会被跳过。用 `embed <路径>` 查看该文件内容，确认是否真的与已有文档完全相同。

### Q3.6：如何重置去重哈希表

```
你的问题> rebuild
```

`rebuild` 会删除整个 `./storage`（包括 docstore.json 中的哈希映射），从头开始构建。

### Q3.7：旧索引（升级前构建的）能直接用去重吗

**不能直接生效**。旧版代码把 chunk 节点存入 docstore（哈希是 chunk 级的），而 IngestionPipeline 做的是**文档级**哈希去重（整个文件内容的 SHA256）。两者哈希粒度不同，旧索引中重新 ingest 已有文件不会被识别为重复。

**解决方案**：执行一次 `rebuild`，用新代码重建索引。重建后 docstore 会记录文档级哈希，后续 ingest 即可正确去重。

```
你的问题> rebuild
```

重建后用 `dedup_status` 验证：

```
你的问题> dedup_status
```

应看到"唯一哈希数"等于去重后的文档数。

### Q3.8：查询报错 `ValueError: setting an array element with a sequence ... inhomogeneous shape`

**原因**：索引中混入了不同维度的向量。最常见场景是：用模型 A（如 bge 512 维）rebuild 了索引，然后换用模型 B（如 MiniLM 384 维）启动并 ingest 了新文档，导致 vector\_store 里同时存在 512 维和 384 维的向量。查询时 NumPy 试图把所有向量堆叠成二维数组，维度不齐就报错。

**已加固**：系统启动时会校验索引向量维度与当前 `EMBED_MODEL` 是否一致，不一致会打印警告并拦截 `ingest`。

**解决**：确保用同一个 `EMBED_MODEL` 启动，然后执行 `rebuild` 重建索引。

### Q3.9：rebuild 报错 `Cannot initialize from a vector store that does not store text`

**原因**：`SimpleVectorStore.stores_text = False`（硬编码类属性），而旧代码用 `VectorStoreIndex.from_vector_store()` 创建索引，该方法要求 `stores_text=True`。

**已修复**：现改为 LlamaIndex 官方推荐模式——pipeline 只做去重+切块+embedding 并返回节点，由 `VectorStoreIndex(nodes=nodes)` 负责写入 vector\_store 和构建 index\_struct。此模式不依赖 `stores_text` 属性。

**升级须知**：若从旧版代码升级，执行一次 `rebuild` 即可。

### Q3.10：在 IDE 内置终端里输出还是会被截断，没看到分页

**原因**：IDE（Trae / VS Code 等）的输出捕获会让 `sys.stdin.isatty()` 返回 False。系统识别为非交互环境，自动降级为直接全打印——这是设计行为，避免在无法接收用户按键的环境里强行分页导致刷屏。

**解决**：在独立的 PowerShell / Windows Terminal 窗口里运行 `python run_vector_demo.py`，即可获得 `more.com` 分页体验（空格翻页、q 退出）。

### Q3.8：高亮只显示 `>>>...<<<` 没有颜色

**原因**：重点句高亮会根据 `sys.stdout.isatty()` 判断输出环境。IDE 内置终端、管道重定向、CI 日志等非 tty 环境下，ANSI 颜色码会被原样打印成乱码，因此系统自动降级为 `fallback_prefix`/`fallback_suffix` 包裹的文本标记。

**解决**：在独立的 PowerShell / Windows Terminal 窗口里运行，即可看到 ANSI 颜色高亮效果（样式由 `highlight_config.json` 配置）。

**调整标记符号**：若想改变非 tty 环境下的标记符号，编辑 `highlight_config.json` 的 `fallback_prefix`/`fallback_suffix` 字段。

### Q3.9：换 gte-Qwen2-7B-instruct 后构建索引非常慢

**原因**：7B 参数的 embedding 模型单条 embedding 约需 100-200ms（GPU 推理），1058 文档（约 2.6 万块）总耗时约 30-60 分钟，远慢于 bge-small 的 1 分钟。

**优化建议**：

- 确保用 FP16 加载（默认配置已自动启用）
- 关闭浏览器、其他 AI 工具，避免显存竞争（实测 Chrome + Obsidian + 飞书 + 千问 同时开会占满显存导致 OOM）
- 首次构建后索引会持久化，后续重启无需重建
- **小批量验证**：用 `$env:MAX_DOCS="10"` 只处理 10 个文档，约 30-60 秒完成，验证流程和进度条正常后再去掉限制全量构建
- **显存紧张时调小参数**：在代码中把 `embed_batch_size` 改为 2，`max_length` 改为 2048（约 9-11 GB 显存占用）
- **监控 GPU**：另一终端运行 `nvidia-smi -l 3`，显存使用超过 15 GB 立即 Ctrl+C 中断
- 如果硬件不够，可换 `BAAI/bge-m3`（568M，3GB 显存，质量接近 7B 但速度快 10 倍）

**为什么进度条"卡在 50%"**：IngestionPipeline 的 `Applying transformations` 进度条统计的是两个 transformation 步骤（切块 + 向量化），不是文档数。第 2 步（embedding）是瓶颈且 LlamaIndex 不提供 chunk 级进度，看起来会卡住，但 GPU 实际在全力运行。

### Q3.10：换 Qwen2 模型后报错 "trust\_remote\_code" 或 "pooling" 错误

**原因**：Qwen2 系列 embedding 模型需要特殊配置（trust\_remote\_code、pooling="last"、FP16 等），代码已经默认处理，但如果环境变量 EMBED\_MODEL 设为别的 Qwen2 变体（如 gte-Qwen2-1.5B-instruct），需要确保这些参数都正确。

**检查清单**：

- `trust_remote_code=True`（必须）
- `pooling="last"`（必须）
- `model_kwargs={"torch_dtype": torch.float16}`（必须，否则爆显存）
- `query_instruction` 前缀格式正确（`Instruct: ...\nQuery: `   ）

代码已通过 `_IS_QWEN2_EMBED` 自动检测模型名包含 "qwen2" 或 "gte-Qwen" 时应用上述配置，无需手动修改。

### Q4：查询结果与笔记无关

可能原因：

1. embedding 模型不适合中文 → 换 `Alibaba-NLP/gte-Qwen2-7B-instruct`（默认）或 `BAAI/bge-m3`
2. top\_k 太小 → 改代码第 85 行加 `similarity_top_k=5`
3. 索引过期 → `rebuild` 重建
4. 换个问法，用笔记中可能出现的原词

### Q5：首次构建很慢

- CPU 版 torch：千级文档约 5 分钟
- GPU 版 torch：千级文档约 30 秒
- 后续启动秒加载（直接读 ./storage）

### Q6：ingest 时卡住不动

看进度条最后一行显示的文件名，是那个文件特别大。耐心等待，或 Ctrl+C 退出后用 `embed` 单独测试该文件。

### Q7：如何完全清空重来

```powershell
你的问题> rebuild
```

或手动删除 `./storage/<当前知识库 ID>` 目录后重启脚本。多知识库模式下，每个库有独立子目录（如 `./storage/beijing_daily`），删除一个不影响其他库。

### Q8：日志太多太吵

编辑 [run\_vector\_demo.py 第 18 行](file:///c:/code/LlamaIndex/run_vector_demo.py#L18)，把 `level=logging.INFO` 改成 `level=logging.WARNING`。

### Q9：启动时没看到数据集选择菜单

当前版本**始终显示选择菜单**（即使只有一个配置）。如果没看到菜单，可能原因：

- `kb_configs/` 目录不存在或没有 `.yaml` 文件（只有 `.yaml.example` 不算）→ 系统会直接进入"新建数据集"流程
- 检查 `kb_configs/` 下是否有 `.yaml` 文件（不是 `.yaml.example`）

如果想在菜单中增加选项，可以：

1. 启动时选 `+ 新建数据集` 交互式创建
2. 或手动复制 `.yaml.example` 为 `.yaml` 并修改

### Q10：如何切换到另一个知识库

启动前选定后，**会话期间锁定**，无法运行时切换。请执行：

```
你的问题> exit
```

然后重新启动 `python run_vector_demo.py`，在菜单中选择目标知识库。这是为了确保不同知识库的内容互不污染。

### Q11：新增了 `.yaml` 但菜单里没出现

- 确认文件扩展名是 `.yaml` 而不是 `.yaml.example`（后者不会被加载）
- 确认 YAML 格式正确（`name` 和 `data_dir` 字段必填）
- 用 `kbs` 命令在会话内查看已加载的知识库列表

***

## 十、项目文件结构

```
LlamaIndex/
├── run_vector_demo.py        # 主脚本（CLI 交互式 RAG 问答）
├── app.py                    # Streamlit Web UI（浏览器界面，复用主脚本函数）
├── generate_kb_configs.py    # 批量生成知识库 YAML 配置脚本
├── highlight_config.json     # 高亮样式配置（颜色/下划线/斜体等）
├── MANUAL.md                 # 本手册
├── PROJECT_OVERVIEW.md       # 项目结构解析和功能摘要文档
├── newspaper_analysis_report.md  # 36 份报纸支持度分析报告
├── requirements.txt          # 依赖清单
├── .venv/                    # 虚拟环境（不打包，部署时重建）
├── kb_configs/               # 多知识库 YAML 配置目录（36 个）
│   ├── beijing_daily.yaml         # 北京日报知识库（已构建索引）
│   ├── china_consumer_news.yaml   # 中国消费者报
│   ├── ...                        # 共 36 份报纸配置
│   ├── beijing_daily.yaml.example # 配置模板（不生效）
│   ├── literature.yaml.example    # 文学示例模板
│   └── history.yaml.example       # 历史示例模板
├── models/                   # 本地 embedding 模型（统一管理，便于打包）
│   ├── bge-m3/                    # 2.2 GB（默认推荐）
│   ├── bge-small-zh-v1.5/         # 92 MB
│   ├── all-MiniLM-L6-v2/          # 87 MB
│   └── gte-Qwen2-7B-instruct/     # 29 GB
├── storage/                  # 索引持久化目录（每个知识库独立子目录）
│   └── beijing_daily/             # 北京日报的索引（rebuild 会清空此子目录）
│       ├── docstore.json         # 文档存储 + 哈希映射（去重依据）
│       ├── index_store.json      # 索引元数据
│       └── vector_store.json     # 向量数据
├── _deploy_pkg/              # 部署包暂存目录
│   ├── run_vector_demo.py
│   ├── highlight_config.json # 高亮样式配置（随主脚本一起部署）
│   ├── requirements.txt
│   ├── deploy.ps1
│   └── README_DEPLOY.txt
└── LlamaIndex_RAG_deploy.rar # 打包好的部署包
```

**`docstore.json`** **是去重的关键**：它记录了所有已 ingest 文档的 SHA256 哈希映射。删除它会导致去重记忆丢失（但向量数据仍在，只是无法识别重复）。每个知识库独立一份，互不影响。

**多知识库独立性**：`kb_configs/` 里每个 `.yaml` 对应 `storage/` 下一个独立子目录。切换知识库需重启程序，会话期间锁定一个库，避免内容污染。

***

## 十一、API Key 安全规范

1. **只用环境变量传递**，不写入任何文件
2. **不在聊天/邮件/截图里明文发送**
3. **定期轮换**（每 1-3 个月重置一次）
4. **发现泄露立即吊销**：<https://platform.deepseek.com/api_keys>
5. **部署到新机器后，旧机器的 Key 也建议轮换**

***

## 十二、扩展方向

如需以下功能，需在现有脚本基础上扩展：

| 需求                      | 实现方式                                                      |
| ----------------------- | --------------------------------------------------------- |
| **多知识库独立管理**            | ✅ 已实现：`kb_configs/*.yaml` + 启动前单选 + `storage/<kb_id>/` 隔离 |
| **BGE-M3 默认 embedding** | ✅ 已实现：启动时自动扫描本地模型，bge-m3 为默认选项（一直回车即用）                    |
| **语义去重（Phase 2）**       | 基于 embedding 余弦相似度（>0.95）识别"略改的转载"，需在 ingest 后做一轮近邻搜索     |
| 标签化（景色/转折/三字短语）         | ingest 时用 LLM 打标签存 metadata                               |
| 按文件过滤查询                 | `MetadataFilters` 按 file\_path 过滤                         |
| 更大规模数据（10万+ chunk）      | 换 FAISS/Milvus/Chroma 向量库                                 |
| Obsidian 内右键 ingest     | 开发 Obsidian 插件 + 本地 HTTP 服务                               |
| 本地 LLM（不用 DeepSeek）     | 装 Ollama，换 `llama-index-llms-ollama`                      |
| 多用户并发                   | 改成 FastAPI 服务，索引常驻内存                                      |
| 运行时热切换知识库               | 改成服务化架构，每次请求带 kb\_id 参数（注意隔离）                             |

### 去重路线图

| 阶段      | 方案                     | 状态     | 说明                         |
| ------- | ---------------------- | ------ | -------------------------- |
| Phase 1 | 哈希去重（DUPLICATES\_ONLY） | ✅ 已实现  | 内容完全相同的文档自动跳过，零误判          |
| Phase 2 | 语义去重（embedding 相似度）    | 📋 规划中 | 识别"略改的转载"，基于余弦相似度阈值        |
| Phase 3 | 段落级去重                  | 📋 规划中 | 切块后对 chunk 做相似度去重，处理"部分转载" |

