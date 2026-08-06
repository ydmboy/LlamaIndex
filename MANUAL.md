# LlamaIndex RAG 问答系统 — 使用手册

基于 LlamaIndex 的交互式 RAG（检索增强生成）问答系统。本地 bge-m3 embedding + 云端 LLM（默认 DeepSeek），支持多知识库隔离、哈希去重、三种检索模式，提供 CLI 和 Web UI 两种界面。

> 本手册基于 2026-07-23 在本机（RTX 5060 / Windows）的实测重写，所有菜单文本、命令、输出均经过真实运行验证。

---

## 一、架构与工作流程

```
启动 → 选数据库（kb_configs/*.yaml）→ 选 LLM → 选 embedding → 加载索引 → 选检索模式
       ↓ 会话期间锁定一个库，ingest/query/rebuild 只作用于此库

库内数据流：
文档(.md/.txt)
  → SimpleDirectoryReader 读取（filename_as_id=True）
  → IngestionPipeline（SHA256 文档级哈希去重，重复跳过）
  → SentenceSplitter 切块（chunk=512, overlap=50）
  → HuggingFaceEmbedding 向量化（本地 bge-m3，FP16/GPU）
  → QdrantVectorStore（server 模式：本机 qdrant.exe 由程序自动拉起，on_disk mmap）+ SQLiteKVStore（自研）持久化到 ./storage/<kb_id>/
```

### 核心组件

| 组件        | 配置                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------ |
| LLM       | 启动时五选一：DeepSeek（默认）/ 通义千问 / 智谱 GLM / Ollama / 自定义 OpenAI 兼容                                            |
| Embedding | `models/bge-m3`（1024 维，FP16，GPU），启动时可改选                                                                |
| 切块        | `SentenceSplitter` chunk_size=512, overlap=50                                                          |
| 向量库       | `QdrantVectorStore`（server 模式：Docker 容器 `llamaindex-qdrant` 由程序自动拉起，仅绑定 127.0.0.1）：真正的 HNSW 索引、增量写入、metadata 过滤、支持删除；向量/payload on_disk（mmap 按需读盘），大库启动秒级、常驻内存 1–2GB |
| 文档库/索引库   | `KVDocumentStore` + `KVIndexStore`，底层为项目自研 `SQLiteKVStore`（`sqlite_kvstore.py`，官方无 SQLite 包）：按需查询、增量提交 |
| 去重        | `IngestionPipeline` + `DocstoreStrategy.DUPLICATES_ONLY`（文档级 SHA256）                                   |
| 检索        | 向量检索（默认）/ 聚合直遍 / 全文搜索（BM25），运行中 `mode` 切换                                                              |
| 高亮        | 检索片段内自动高亮重点句（本地 embedding 余弦相似度，纯展示层）                                                                  |

---

## 二、环境与本机部署状态

### 本机已部署完成（2026-07-23 实测通过，2026-07-29 更新 Faiss 后端）

- `.venv`：Python 3.11.15
- `torch 2.7.1+cu128`：**注意不是手册旧版写的 2.5.1+cu121**。RTX 5060 是 Blackwell 架构（sm_120），cu121 版实测报 `no kernel image is available`，无法运算；cu128 版已实测 FP16 GPU 矩阵运算正常
- `qdrant-client 1.18.0`（Python 客户端）+ Docker 容器 `qdrant/qdrant:v1.19.0`（服务端，容器名 `llamaindex-qdrant`，程序自动拉起）：向量存储后端（2026-08-06 起从 local 内嵌模式切换到 server 模式，解决 184 万点启动全量加载 15+ 分钟的问题。注意：qdrant 官方 Windows 裸 exe 实测有删除/段合并 rename 失败 bug #5924 导致段泄漏，不可用，必须用 Docker 容器）
- `models/bge-m3/`：2.27 GB（主权重为 `pytorch_model.bin`，该仓库**没有** safetensors 文件）
- `requirements.txt` 全部依赖已安装，版本与清单一致

> **不要重跑 `_deploy_pkg/deploy.ps1`**：它会装回 torch 2.5.1+cu121（在本机 GPU 上不可用），且它检查 `model.safetensors` 是否存在的逻辑永远不会命中（bge-m3 仓库无此文件），会导致重复下载。

### 新机器手动部署（家中电脑照此逐步执行）

**第 0 步：装三个前置软件**
- Python 3.11（3.10 亦可，勾"Add to PATH"）：https://www.python.org/downloads/
- Git：https://git-scm.com/download/win
- Docker Desktop：https://www.docker.com/products/docker-desktop/
  - 安装后启动一次，确认能正常运行（首次启动较慢）
  - 建议 Settings → General 勾选 "Start Docker Desktop when you sign in"（登录自启，之后无感）
  - 若启动报 "Virtualization support not detected"：用**管理员 PowerShell** 执行 `wsl --install --no-distribution`，然后**重启电脑**

**第 1 步：拉代码 + 建虚拟环境**

```powershell
git clone <仓库地址> LlamaIndex
cd LlamaIndex
python -m pip install uv
uv venv .venv --python 3.11
```

**第 2 步：装 torch（按显卡选一行）**

```powershell
# RTX 50 系（Blackwell，必须 cu128）
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# 老 N 卡（RTX 20/30/40 系）
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# 无 NVIDIA 显卡（CPU 版，embedding 会慢 5-10 倍）
uv pip install torch==2.5.1
```

**第 3 步：装其余依赖 + 下载 embedding 模型（约 4.25GB）**

```powershell
uv pip install -r requirements.txt
.venv\Scripts\python.exe download_models.py --model bge-m3
```

**第 4 步：验证 GPU 可用**

```powershell
.venv\Scripts\python.exe -c "import torch; x=torch.randn(512,512,device='cuda'); print('GPU OK:', float((x@x).sum()))"
```

**第 5 步：设置环境变量并启动**

```powershell
$env:DEEPSEEK_API_KEY="sk-你的key"      # 每个新终端都要重设
$env:LLM_PROVIDER="deepseek"
.venv\Scripts\python.exe run_vector_demo.py
```

首次启动时程序会自动拉取 `qdrant/qdrant` 镜像（约 100MB）并创建 `llamaindex-qdrant` 容器——看到 `[Qdrant] 服务端就绪` 即成功。之后选数据库 → `ingest <路径>` 添加文档 → 提问。

**（可选）第 6 步：从旧机器迁移已有数据，不重跑 embedding**

1. 旧机器上拷贝整个 `storage/<库id>/` 目录（含 `docstore.db` 和 `qdrant/`）到新机器项目相同位置；`kb_configs/*.yaml` 也一并拷贝
2. 新机器上执行一次性迁移脚本（把 local 数据搬进 Docker qdrant）：
   ```powershell
   .venv\Scripts\python.exe migrate_qdrant_to_server.py
   ```
   跑完自动校验点数；确认程序查询正常后，可删除 `storage/<库id>/qdrant/` 释放空间（`docstore.db` 要保留）
3. 不迁移也可以：`docstore.db` 保留去重记忆，`rebuild` 后重新 ingest 源文件

### 验证部署

```powershell
.venv\Scripts\python.exe -c "import torch; x=torch.randn(512,512,device='cuda'); print('GPU OK:', float((x@x).sum()))"
```

只查 `torch.cuda.is_available()` 不够——cu121 在 RTX 5060 上它也返回 True，但实际运算会报错，必须做一次真实矩阵运算。

---

## 三、启动

```powershell
# 必需（按你选的 LLM 提供商设置对应 Key，只走环境变量）
$env:DEEPSEEK_API_KEY="sk-你的key"

.venv\Scripts\python.exe run_vector_demo.py
```

### 启动交互（共 5 步，一直回车 = 全默认）

1. **选数据库**（会话期间锁定，切换需重启）：

```
============================================================
 请选择要进入的数据库（不同数据库彼此独立，避免内容污染）
============================================================
  1. 代码知识数据库  [code] [空库]
  2. 通用知识数据库  [knowledge] [空库]
  3. 报纸数据库 (默认)  [newspaper] [空库]
  4. + 新建数据库
输入序号 (1-4)，默认 3:
```

2. **选 LLM 提供商**（默认 1 DeepSeek；设 `LLM_PROVIDER` 环境变量可跳过）
3. **选 embedding 模型**（默认本地 `models/bge-m3`；选 0 可手动输入 HF 模型名或路径；设 `EMBED_MODEL` 可跳过）
4. **加载索引**：已构建的库直接加载（秒级）；空库自动创建空索引并提示用 `ingest` 添加文档
5. **选检索模式**（默认 1 向量检索，运行中可用 `mode` 随时切换）

**一直回车** = 报纸数据库 + DeepSeek + bge-m3 + 向量检索。

### 新建数据库（菜单第 4 项）

实测交互（依次输入）：数据库 ID → 显示名 → 描述（可空）→ 文件扩展名（默认 `.md`）。**不询问数据源目录**——新建的是空库，文档全靠后续 `ingest` 添加。保存为 `kb_configs/<id>.yaml` 并自动进入该库。

### 知识库配置文件

`kb_configs/<kb_id>.yaml`，示例（当前生效的三个库均为此形式）：

```yaml
name: "报纸数据库"
description: "各类报纸全文数据，支持跨报纸检索"
file_exts: [".md", ".txt"]
```

- `*.yaml.example` 不会被加载，只有 `*.yaml` 生效
- 每库索引独立存于 `storage/<kb_id>/`，包含以下内容：
  - `docstore.db` — SQLite 单文件：文档/节点内容 + SHA256 哈希（去重关键）+ 索引结构
  - `simhash_fp.json` — SimHash 转载指纹库
- 向量数据在 Docker named volume `llamaindex-qdrant-data` 中（collection 名 = 库 id；由容器 `llamaindex-qdrant` 托管，on_disk mmap）；旧 local 模式的 `storage/<kb_id>/qdrant/` 目录迁移验证通过后可手动删除
- `storage_dir` 字段留空则自动用 `./storage/<kb_id>`；`data_dir` 可不填（空库靠 ingest）

---

## 四、命令一览（全部实测）

在 `[模式] 你的问题>` 提示符后输入。直接输入任何问题即按当前模式查询。

| 命令（别名）                         | 说明                                           |
| ------------------------------ | -------------------------------------------- |
| `ingest <路径>`（`加入`）            | 把文件/文件夹增量加入当前库，自动哈希去重                        |
| `embed <路径>`                   | 向量化单个文件并打印结果，不写入索引（调试用）                      |
| `list`（`列表`/`ls`）              | 列出当前库已 ingest 的文件路径                          |
| `dedup_status`（`dedup`/`去重统计`） | 去重统计：唯一哈希数/文档节点数/向量数/容量                      |
| `mode`（`模式`/`检索模式`）            | 切换检索模式（仅弹出菜单选序号，**不支持** `mode vector` 带参数形式） |
| `kbs`（`知识库`/`kb`）              | 查看所有库状态（容量/节点数/当前标记），切换需重启                   |
| `rebuild`                      | **无确认提示**，立即清空当前库索引并重建空索引                    |
| `clear`（`cls`）                 | 清屏                                           |
| `exit`（`quit`/`退出`/`q`）        | 退出                                           |

### ingest 示例（实测输出）

```
[ingest] 目标: D:\docs -> 知识库 [报纸数据库]
    -> 加载 3 个文档，开始增量写入（哈希去重）...
    [1/3] 处理: 北京日报-2026-07-01.md（产出 1 节点）
    -> 完成！新增 3 个文档（3 个节点），跳过 0 个重复
    -> 已持久化到 ./storage/newspaper，docstore 共跟踪 6 个唯一哈希
```

- 重复 ingest 同一批文件：`新增 0，跳过 N 个重复`，秒级完成
- 「唯一哈希」统计的是**文档 + 切块节点**的总数（3 个文件各产 1 节点 → 6 个哈希），不是单纯文件数

### rebuild 注意

无确认、立即删除 `storage/<当前库>` 并重建空索引（server 模式下会同时删除 qdrant 服务端的对应 collection）。换 embedding 模型后必须 rebuild（维度不同，系统启动时会校验并拦截不一致的 ingest）。

---

## 五、检索模式（实测）

| 模式           | 原理                                       | 是否调 LLM        | 适用场景    |
| ------------ | ---------------------------------------- | -------------- | ------- |
| **向量检索**（默认） | 查询向量化 → 余弦相似度 top_k=10 → LLM 生成回答 + 来源片段 | 是（需有效 API Key） | 语义问答    |
| **聚合直遍**     | 遍历 docstore 全部节点，按规则提取匹配行，100% 覆盖        | 否              | 列举类查询   |
| **全文搜索**     | 中文 bigram 分词 + BM25 排序 top_k=20 + 关键词高亮  | 否              | 关键词精确匹配 |

- **向量检索**：`similarity_top_k` 在 `retrieval_config.yaml` 统一配置（当前值 10，代码兜底默认 5）。枚举类问题覆盖率低，此类问题请用聚合模式
- **聚合直遍**三种提取策略（按查询词自动选择）：
  - 含「标题/文章名/文章列表/文章目录」→ 提取 `### ` 开头的 markdown 标题行
  - 含「项目」→ 提取包含「项目」的行
  - 其他 → 提取包含关键词的行（去掉"列出所有"等虚词）
- **全文搜索**：倒排索引首次使用时懒构建，之后复用；分词示例：查询"量子计算"→ `量子 / 子计 / 计算`
- **重点句高亮**：向量模式的来源片段会自动高亮最相关的 1-2 句（阈值 0.5）。真实终端显示 ANSI 彩色（样式见 `highlight_config.json`）；IDE/管道环境降级为 `>>>重点句<<<` 文本标记——这是设计行为，不是故障
- **长输出分页**：超过 30 行自动用系统分页器（空格翻页、q 退出）；IDE 内置终端自动改为全打印

---

## 六、Web UI（Streamlit，实测 HTTP 200）

```powershell
$env:DEEPSEEK_API_KEY="sk-你的key"
$env:STREAMLIT_HOME="c:\code\LlamaIndex\.streamlit"
.venv\Scripts\streamlit.exe run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

访问 http://localhost:8501 。功能：侧边栏选库/切模式、💬 问答、📥 Ingest（目录导入/文件上传）、🗂️ 库管理。

注意：Web UI 只能加载**已构建索引**的库；`--browser.gatherUsageStats false` 必须加（避免写入无权限目录）。

---

## 七、配置项

### 环境变量

| 变量                                    | 必需           | 说明                                                        |
| ------------------------------------- | ------------ | --------------------------------------------------------- |
| `DEEPSEEK_API_KEY`                    | 选 DeepSeek 时 | DeepSeek API Key                                          |
| `DEEPSEEK_MODEL`                      | 否            | DeepSeek 模型名（默认 `deepseek-v4-pro`，可选 `deepseek-v4-flash`） |
| `DASHSCOPE_API_KEY` / `ZHIPU_API_KEY` | 选对应厂商时       | 千问 / 智谱 Key                                               |
| `LLM_PROVIDER`                        | 否            | `deepseek`/`qwen`/`zhipu`/`ollama`/`custom`，预设后跳过 LLM 菜单  |
| `EMBED_MODEL`                         | 否            | embedding 模型路径或 HF ID，预设后跳过模型菜单                           |
| `DATA_DIR` / `STORAGE_DIR`            | 否            | 仅无 `kb_configs` 时的遗留单库模式用，当前三库配置下用不到                      |
| `QDRANT_MODE`                         | 否            | `server`（默认，自动拉起 Docker 容器）/ `local`（回退旧内嵌模式）             |
| `QDRANT_HOST` / `QDRANT_PORT`         | 否            | qdrant 服务端绑定地址（默认 `127.0.0.1:16333`，只绑定回环；不用 6333 是因为它落在 Windows Hyper-V 动态端口排除段里） |
| `QDRANT_CONTAINER` / `QDRANT_IMAGE` / `QDRANT_VOLUME` | 否 | qdrant 容器名 / 镜像 / 数据卷（默认 `llamaindex-qdrant` / `qdrant/qdrant:v1.19.0` / `llamaindex-qdrant-data`） |
| `MAX_DOCS`                            | 否            | 调试用，限制上述遗留模式构建时的文档数                                       |

### 检索参数配置（`retrieval_config.yaml`）

集中管理所有检索参数，CLI/Web 共用一份。文件不存在时用代码内默认值，程序可独立运行。修改后重启生效。

| 分区          | 参数                      | 默认值              | 说明                                               |
| ----------- | ----------------------- | ---------------- | ------------------------------------------------ |
| `vector`    | `similarity_top_k`      | 5                | 向量检索返回片段数（CLI/Web 统一）                            |
| `vector`    | `response_mode`         | `tree_summarize` | LLM 回答模式                                         |
| `fulltext`  | `top_k`                 | 20               | 全文搜索返回片段数                                        |
| `fulltext`  | `bm25_k1`               | 1.5              | BM25 词频饱和参数                                      |
| `fulltext`  | `bm25_b`                | 0.75             | BM25 文档长度归一化参数                                   |
| `chunk`     | `chunk_size`            | 512              | 文本分块大小                                           |
| `chunk`     | `chunk_overlap`         | 50               | 分块重叠                                             |
| `highlight` | `top_n`                 | 2                | 每个片段高亮几句                                         |
| `highlight` | `threshold`             | 0.5              | 相似度低于此值的句子不高亮                                    |
| `ingest`    | `batch_size`            | 200              | 落盘检查点间隔：每 N 个文件持久化一次，中断后重跑自动跳过已落盘文件              |
| `ingest`    | `doc_batch_size`        | 32               | 向量化小批大小：每积累 N 个文件统一跑一次 pipeline，让 embedding 吃满批量 |
| `ingest`    | `auto_continue_timeout` | 10               | 每批结束后等待确认的秒数，超时自动继续                              |
| `dedup`     | `simhash_enabled`       | `true`           | SimHash 近似转载拦截开关（报纸转载/不同文件同内容时跳过）                |
| `dedup`     | `simhash_threshold`     | 3                | 64 位指纹汉明距离阈值，≤ 此值判为同一篇（调大更宽松、误杀增多）               |
| `dedup`     | `simhash_min_chars`     | 8                | 正文归一化后短于此长度只按精确哈希去重，不算 SimHash                   |

> 注：表中"默认值"指 YAML 文件缺失时的代码兜底值。当前仓库 `retrieval_config.yaml` 实际配置为 `similarity_top_k: 10`、`fulltext.top_k: 20`，以 YAML 为准。

### 代码常量（`run_vector_demo.py` 顶部）

| 常量                                        | 值                             | 说明                |
| ----------------------------------------- | ----------------------------- | ----------------- |
| `KB_CONFIGS_DIR`                          | `kb_configs`                  | 库配置目录             |
| `DEFAULT_KB_ID`                           | `newspaper`                   | 启动默认库             |
| `CHUNK_SIZE` / `CHUNK_OVERLAP`            | 由 `retrieval_config.yaml` 控制  | 切块参数（默认 512/50）   |
| `HIGHLIGHT_TOP_N` / `HIGHLIGHT_THRESHOLD` | 由 `retrieval_config.yaml` 控制  | 重点句高亮参数（默认 2/0.5） |
| `PAGER_THRESHOLD_LINES`                   | 30                            | 超过此行数触发分页         |
| `FAISS_INDEX_FILENAME`                    | `default__vector_store.faiss` | Faiss 索引文件名       |

### Qdrant server 模式参数（`_make_storage_context` / `_ensure_qdrant_server` 函数）

| 参数                 | 值                              | 说明                                       |
| ------------------ | ------------------------------ | ---------------------------------------- |
| 监听地址/端口            | `127.0.0.1:16333`              | 只绑定回环：不弹防火墙、局域网不可达（6333 被 Windows 端口排除段占用，故用 16333） |
| 容器 / 镜像            | `llamaindex-qdrant` / `qdrant/qdrant:v1.19.0` | Docker 托管，程序自动拉起；容器常驻（unless-stopped） |
| 数据卷                | `llamaindex-qdrant-data`（named volume） | 所有库共用一个 server，collection 名 = 库 id      |
| `on_disk`（向量）      | `true`                         | 向量 mmap 按需读盘，大库启动秒级、常驻内存低的关键             |
| `on_disk_payload`  | `true`                         | payload 同上                               |
| 距离度量 / 维度          | `Cosine` / 随 embedding 模型      | 预创建 collection 时按当前模型维度                  |

> 相关环境变量：`QDRANT_MODE`（`server` 默认 / `local` 回退内嵌模式）、`QDRANT_HOST`、`QDRANT_PORT`、`QDRANT_CONTAINER`、`QDRANT_IMAGE`、`QDRANT_VOLUME`。
> 容器由程序自动管理（Docker Desktop 未运行会先拉起）；想手动操作：`docker logs llamaindex-qdrant` / `docker stop llamaindex-qdrant`。
> 切勿改用 qdrant 官方 Windows 裸 exe——其删除/段合并 rename 有 bug（[#5924](https://github.com/qdrant/qdrant/issues/5924)），写入负载下段文件泄漏会耗尽磁盘（本机实测）。

### 高亮样式（`highlight_config.json`）

字段：`foreground`/`background`（标准色或亮色名，null 表示不设）、`bold`/`italic`/`underline`/`strikethrough`/`reverse`（布尔）、`fallback_prefix`/`fallback_suffix`（非 tty 环境的文本标记，默认 `>>>`/`<<<`）。文件缺失或格式错误时用内置默认（亮黄+下划线）。

---

## 八、原理与去重

### 数据流

1. `SimpleDirectoryReader` 读文件成 Document（不解析 markdown 结构）
2. `IngestionPipeline` 算文档 SHA256，重复直接跳过
3. `SentenceSplitter` 按句子边界切块
4. bge-m3 把 chunk 转成 1024 维向量
5. `QdrantVectorStore`（server 模式 HNSW）图检索，O(log N) 近似最近邻
6. 按 `similarity_top_k`（当前 10）取片段交 LLM 生成回答
7. 展示层：本地 embedding 在片段内找 top-N 相似句高亮（不改变检索结果）

### 哈希去重

- **文档级**：一个文件 = 一个文档，整体 SHA256。内容完全相同的文件才会跳过；部分重叠（转载）不算重复，会都入库
- 哈希映射持久化在 `docstore.db`（自研 SQLiteKVStore），重启后仍记得
- 向量数据在 qdrant server 的共享存储 `storage/_qdrant_server/`（collection 名 = 库 id），与 docstore 分离
- 不会做的事：不标签化、不分类、不解析结构、不语义去重（"略改的转载"识别不了）

---

## 九、常见问题

**Q：报错"请先设置环境变量 DEEPSEEK_API_KEY"**
每个新终端都要重新执行 `$env:DEEPSEEK_API_KEY="sk-..."`（环境变量不持久）。

**Q：只有向量模式没反应/报错？**
向量模式的回答生成需要有效的 LLM API Key 且能访问对应 API。聚合和全文模式不依赖 LLM，可无网使用。

**Q：大批量 ingest 越到后越慢 / 最后写盘时 MemoryError？**
已随存储后端更换（2026-07-30，Qdrant + SQLite；2026-08-06 起 Qdrant 改 server 模式）从根上解决：① 向量库 Qdrant server 模式，增量写入、on_disk mmap，不再全量重写索引、启动不再全量载入内存；② docstore/index_store 换自研 SQLiteKVStore（`sqlite_kvstore.py`），原文和哈希按需查询、增量提交，启动不再全量解析 JSON；③ 哈希缓存，去重检查 O(1)；④ 逐文件惰性读取，不全量加载；⑤ 小批向量化，每积累 `ingest.doc_batch_size`（默认 32）个文件统一跑一次 pipeline。崩溃安全由 Qdrant WAL + SQLite 事务保证，写盘途中断电/强杀不会留下截断文件。此外 **ingest 每 200 个文件自动落盘一个检查点**（`retrieval_config.yaml` 的 `ingest.batch_size` 可调），中断/崩溃后重新执行同一 ingest 命令，哈希去重自动跳过已落盘文件，断点续传。**中断方式**：批次边界输入 `q` 退出，或直接 Ctrl+C——后者会被捕获并把已处理部分落盘后回到提示符，不再崩溃退出。

**Q：旧索引（Faiss / SimpleVectorStore 格式）如何迁移到 Qdrant？**
不兼容直接迁移，也无迁移工具——2026-07-30 换后端时已决定存量数据不保留。直接 `rebuild` 后重新 ingest 源文件即可。

**Q：Qdrant local 内嵌模式的旧数据如何迁移到 server 模式？**
用一次性脚本 `.venv/Scripts/python.exe migrate_qdrant_to_server.py`：直接读取 `storage/<库>/qdrant/` 里的向量并原样写入 server（约 184 万点 30–60 分钟，**不重新 embedding**），支持断点续传，跑完自动校验点数和抽样 payload。验证程序正常后手动删除 `storage/<库>/qdrant/` 释放空间即可。

**Q：RTX 50 系显卡报 `no kernel image is available for execution on the device`**
torch 版本太旧（sm_120 需要 cu128 构建）：`uv pip install --reinstall-package torch torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`

**Q：换 embedding 模型后 ingest 被拦截/查询报错**
不同模型向量维度不同，旧索引不兼容。启动时的维度校验会拦截 ingest，执行 `rebuild` 重建即可。

**Q：高亮只显示 `>>>...<<<` 没有颜色 / 没看到分页**
IDE 内置终端和管道环境是非 tty，系统自动降级：高亮变文本标记、分页变全打印。在独立 PowerShell / Windows Terminal 运行即有彩色高亮和分页。

**Q：如何完全清空某个库**
`rebuild`（无确认，立即执行），或手动删除 `storage/<库id>` 目录后重启（server 模式下程序会自动清理 qdrant 服务端的对应 collection）。

**Q：下载模型失败 / hf-mirror 镜像**
hf-mirror.com 目前已失效（302 跳回 huggingface.co）。本机实测可直连 huggingface.co；不行的话用 `HF_HUB_DISABLE_XET=1` 环境变量再试，或手动 curl 下载权重到 `models/bge-m3/`。

**Q：bge-m3 目录里没有 model.safetensors**
正常。该仓库主权重就是 `pytorch_model.bin`（2.27GB），sentence-transformers 直接加载它。

**Q：启动没看到数据库选择菜单**
检查 `kb_configs/` 下是否有 `.yaml` 文件（`.yaml.example` 不算）。没有任何 yaml 时会直接进入新建流程。

---

## 十、项目文件结构

```
LlamaIndex/
├── run_vector_demo.py        # 主脚本（CLI 交互式 RAG 问答）
├── app.py                    # Streamlit Web UI
├── sqlite_kvstore.py         # 自研 SQLiteKVStore（docstore/index_store 后端，官方无此包）
├── download_models.py        # 下载 embedding 模型到 models/
├── generate_kb_configs.py    # 批量生成报纸库配置的历史脚本（当前用不到）
├── analyze_newspapers.py     # 报纸数据分析脚本（工具）
├── test_modes.py / test_qwen2_embed.py / test_topk.py  # 开发测试脚本
├── highlight_config.json     # 高亮样式配置
├── requirements.txt          # 依赖清单（不含 torch）
├── MANUAL.md                 # 本手册
├── PROJECT_OVERVIEW.md       # 项目结构解析（历史文档，部分过时）
├── kb_configs/               # 库配置：code / knowledge / newspaper 三个生效
│   └── *.yaml.example        # 模板（不生效）
├── models/bge-m3/            # 本地 embedding 模型（2.27GB）
├── migrate_qdrant_to_server.py # local→server 一次性迁移脚本（已执行，存档备查）
├── storage/<kb_id>/          # 各库索引（docstore.db SQLite + simhash_fp.json）
│                             # 向量数据在 Docker volume llamaindex-qdrant-data（collection 名 = 库 id）
└── _deploy_pkg/              # 部署包（deploy.ps1 等；torch 版本与模型检查已过时，勿直接重跑）
```

`docstore.db`（SQLite）是去重的关键：记录所有已 ingest 文档/节点的 SHA256 哈希和原文。删除它会导致去重记忆丢失。向量数据在 Docker named volume `llamaindex-qdrant-data`（qdrant 容器托管，HNSW 索引 + WAL，collection 名 = 库 id）；旧 local 模式的 `storage/<库>/qdrant/` 迁移验证通过后可删。

---

## 十一、API Key 安全

1. 只用环境变量传递，不写入任何文件、不提交 git
2. 不在聊天/邮件/截图里明文发送
3. 定期轮换，发现泄露立即到 <https://platform.deepseek.com/api_keys> 吊销
