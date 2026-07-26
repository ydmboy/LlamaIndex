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
  → SimpleVectorStore + SimpleDocumentStore 持久化到 ./storage/<kb_id>/
```

### 核心组件

| 组件 | 配置 |
|------|------|
| LLM | 启动时五选一：DeepSeek（默认）/ 通义千问 / 智谱 GLM / Ollama / 自定义 OpenAI 兼容 |
| Embedding | `models/bge-m3`（1024 维，FP16，GPU），启动时可改选 |
| 切块 | `SentenceSplitter` chunk_size=512, overlap=50 |
| 向量库/文档库 | `SimpleVectorStore` + `SimpleDocumentStore`，JSON 持久化，每库独立 |
| 去重 | `IngestionPipeline` + `DocstoreStrategy.DUPLICATES_ONLY`（文档级 SHA256） |
| 检索 | 向量检索（默认）/ 聚合直遍 / 全文搜索（BM25），运行中 `mode` 切换 |
| 高亮 | 检索片段内自动高亮重点句（本地 embedding 余弦相似度，纯展示层） |

---

## 二、环境与本机部署状态

### 本机已部署完成（2026-07-23 实测通过）

- `.venv`：Python 3.11.15
- `torch 2.7.1+cu128`：**注意不是手册旧版写的 2.5.1+cu121**。RTX 5060 是 Blackwell 架构（sm_120），cu121 版实测报 `no kernel image is available`，无法运算；cu128 版已实测 FP16 GPU 矩阵运算正常
- `models/bge-m3/`：2.27 GB（主权重为 `pytorch_model.bin`，该仓库**没有** safetensors 文件）
- `requirements.txt` 全部依赖已安装，版本与清单一致

> **不要重跑 `_deploy_pkg/deploy.ps1`**：它会装回 torch 2.5.1+cu121（在本机 GPU 上不可用），且它检查 `model.safetensors` 是否存在的逻辑永远不会命中（bge-m3 仓库无此文件），会导致重复下载。

### 新机器部署（仅供参考）

```powershell
uv venv .venv --python 3.11
# NVIDIA 显卡（RTX 50 系必须 cu128；老显卡可改用 cu121 源装 torch==2.5.1）
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
# 下载 bge-m3 模型（约 2.3GB；也可 python download_models.py）
uv run --with huggingface-hub python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3', ignore_patterns=['*.h5','*.msgpack','onnx/*','openvino/*'])"
```

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
- 每库索引独立存于 `storage/<kb_id>/`（docstore.json / index_store.json / vector_store.json）
- `storage_dir` 字段留空则自动用 `./storage/<kb_id>`；`data_dir` 可不填（空库靠 ingest）

---

## 四、命令一览（全部实测）

在 `[模式] 你的问题>` 提示符后输入。直接输入任何问题即按当前模式查询。

| 命令（别名） | 说明 |
|---|---|
| `ingest <路径>`（`加入`） | 把文件/文件夹增量加入当前库，自动哈希去重 |
| `embed <路径>` | 向量化单个文件并打印结果，不写入索引（调试用） |
| `list`（`列表`/`ls`） | 列出当前库已 ingest 的文件路径 |
| `dedup_status`（`dedup`/`去重统计`） | 去重统计：唯一哈希数/文档节点数/向量数/容量 |
| `mode`（`模式`/`检索模式`） | 切换检索模式（仅弹出菜单选序号，**不支持** `mode vector` 带参数形式） |
| `kbs`（`知识库`/`kb`） | 查看所有库状态（容量/节点数/当前标记），切换需重启 |
| `rebuild` | **无确认提示**，立即清空当前库索引并重建空索引 |
| `clear`（`cls`） | 清屏 |
| `exit`（`quit`/`退出`/`q`） | 退出 |

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

无确认、立即删除 `storage/<当前库>` 并重建空索引。换 embedding 模型后必须 rebuild（维度不同，系统启动时会校验并拦截不一致的 ingest）。

---

## 五、检索模式（实测）

| 模式 | 原理 | 是否调 LLM | 适用场景 |
|------|------|-----------|---------|
| **向量检索**（默认） | 查询向量化 → 余弦相似度 top_k=2 → LLM 生成回答 + 来源片段 | 是（需有效 API Key） | 语义问答 |
| **聚合直遍** | 遍历 docstore 全部节点，按规则提取匹配行，100% 覆盖 | 否 | 列举类查询 |
| **全文搜索** | 中文 bigram 分词 + BM25 排序 top_k=20 + 关键词高亮 | 否 | 关键词精确匹配 |

- **向量检索**：top_k=2 写死在代码中，枚举类问题覆盖率低（实测输出自报"仅覆盖 0.004%"），此类问题请用聚合模式
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

| 变量 | 必需 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | 选 DeepSeek 时 | DeepSeek API Key |
| `DEEPSEEK_MODEL` | 否 | DeepSeek 模型名（默认 `deepseek-v4-pro`，可选 `deepseek-v4-flash`） |
| `DASHSCOPE_API_KEY` / `ZHIPU_API_KEY` | 选对应厂商时 | 千问 / 智谱 Key |
| `LLM_PROVIDER` | 否 | `deepseek`/`qwen`/`zhipu`/`ollama`/`custom`，预设后跳过 LLM 菜单 |
| `EMBED_MODEL` | 否 | embedding 模型路径或 HF ID，预设后跳过模型菜单 |
| `DATA_DIR` / `STORAGE_DIR` | 否 | 仅无 `kb_configs` 时的遗留单库模式用，当前三库配置下用不到 |
| `MAX_DOCS` | 否 | 调试用，限制上述遗留模式构建时的文档数 |

### 检索参数配置（`retrieval_config.yaml`）

集中管理所有检索参数，CLI/Web 共用一份。文件不存在时用代码内默认值，程序可独立运行。修改后重启生效。

| 分区 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| `vector` | `similarity_top_k` | 5 | 向量检索返回片段数（CLI/Web 统一） |
| `vector` | `response_mode` | `tree_summarize` | LLM 回答模式 |
| `fulltext` | `top_k` | 20 | 全文搜索返回片段数 |
| `fulltext` | `bm25_k1` | 1.5 | BM25 词频饱和参数 |
| `fulltext` | `bm25_b` | 0.75 | BM25 文档长度归一化参数 |
| `chunk` | `chunk_size` | 512 | 文本分块大小 |
| `chunk` | `chunk_overlap` | 50 | 分块重叠 |
| `highlight` | `top_n` | 2 | 每个片段高亮几句 |
| `highlight` | `threshold` | 0.5 | 相似度低于此值的句子不高亮 |

### 代码常量（`run_vector_demo.py` 顶部）

| 常量 | 值 | 说明 |
|------|-----|------|
| `KB_CONFIGS_DIR` | `kb_configs` | 库配置目录 |
| `DEFAULT_KB_ID` | `newspaper` | 启动默认库 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 由 `retrieval_config.yaml` 控制 | 切块参数（默认 512/50） |
| `HIGHLIGHT_TOP_N` / `HIGHLIGHT_THRESHOLD` | 由 `retrieval_config.yaml` 控制 | 重点句高亮参数（默认 2/0.5） |
| `PAGER_THRESHOLD_LINES` | 30 | 超过此行数触发分页 |

### 高亮样式（`highlight_config.json`）

字段：`foreground`/`background`（标准色或亮色名，null 表示不设）、`bold`/`italic`/`underline`/`strikethrough`/`reverse`（布尔）、`fallback_prefix`/`fallback_suffix`（非 tty 环境的文本标记，默认 `>>>`/`<<<`）。文件缺失或格式错误时用内置默认（亮黄+下划线）。

---

## 八、原理与去重

### 数据流

1. `SimpleDirectoryReader` 读文件成 Document（不解析 markdown 结构）
2. `IngestionPipeline` 算文档 SHA256，重复直接跳过
3. `SentenceSplitter` 按句子边界切块
4. bge-m3 把 chunk 转成 1024 维向量
5. `SimpleVectorStore` 暴力余弦检索
6. top_k=2 片段交 LLM 生成回答
7. 展示层：本地 embedding 在片段内找 top-N 相似句高亮（不改变检索结果）

### 哈希去重

- **文档级**：一个文件 = 一个文档，整体 SHA256。内容完全相同的文件才会跳过；部分重叠（转载）不算重复，会都入库
- 哈希映射持久化在 `docstore.json`，重启后仍记得
- 不会做的事：不标签化、不分类、不解析结构、不语义去重（"略改的转载"识别不了）

---

## 九、常见问题

**Q：报错"请先设置环境变量 DEEPSEEK_API_KEY"**
每个新终端都要重新执行 `$env:DEEPSEEK_API_KEY="sk-..."`（环境变量不持久）。

**Q：只有向量模式没反应/报错？**
向量模式的回答生成需要有效的 LLM API Key 且能访问对应 API。聚合和全文模式不依赖 LLM，可无网使用。

**Q：大批量 ingest 越到后来越慢？**
旧版每处理一个文件都全量重建一次去重哈希表（总开销 O(N²)），且一次性把全部语料读入内存，几千文件后明显变慢、内存随文件数膨胀。已在 `run_vector_demo.py` 的 ingest 流程中修复：哈希缓存使每次检查降为 O(1)，并改为逐文件惰性读取。修复后每文件耗时不随库增长。**注意**：ingest 循环结束后才统一写索引和持久化，中途 Ctrl+C 不会落盘任何进度；大批量导入时让进程一次性跑完，期间避免运行其他吃内存的程序。

**Q：RTX 50 系显卡报 `no kernel image is available for execution on the device`**
torch 版本太旧（sm_120 需要 cu128 构建）：`uv pip install --reinstall-package torch torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`

**Q：换 embedding 模型后 ingest 被拦截/查询报错**
不同模型向量维度不同，旧索引不兼容。启动时的维度校验会拦截 ingest，执行 `rebuild` 重建即可。

**Q：高亮只显示 `>>>...<<<` 没有颜色 / 没看到分页**
IDE 内置终端和管道环境是非 tty，系统自动降级：高亮变文本标记、分页变全打印。在独立 PowerShell / Windows Terminal 运行即有彩色高亮和分页。

**Q：如何完全清空某个库**
`rebuild`（无确认，立即执行），或手动删除 `storage/<库id>` 目录后重启。

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
├── storage/<kb_id>/          # 各库索引（docstore.json / index_store.json / vector_store.json）
└── _deploy_pkg/              # 部署包（deploy.ps1 等；torch 版本与模型检查已过时，勿直接重跑）
```

`docstore.json` 是去重的关键：记录所有已 ingest 文档/节点的 SHA256 哈希。删除它会导致去重记忆丢失。

---

## 十一、API Key 安全

1. 只用环境变量传递，不写入任何文件、不提交 git
2. 不在聊天/邮件/截图里明文发送
3. 定期轮换，发现泄露立即到 <https://platform.deepseek.com/api_keys> 吊销
