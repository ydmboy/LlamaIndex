# LlamaIndex RAG 知识库系统

基于 LlamaIndex 的本地 RAG 问答系统：多知识库管理、三种检索模式（向量 / 聚合直遍 / 全文 BM25）、CLI + Streamlit Web 双界面、增量 ingest 带哈希去重与断点续传。

## 架构速览

```
文档(.md/.txt)
  → SimpleDirectoryReader 读取
  → IngestionPipeline（SHA256 哈希去重）
  → SentenceSplitter 切块（512/50）
  → bge-m3 向量化（本地模型，FP16/GPU）
  → Qdrant 服务端（Docker 容器，on_disk mmap，启动秒级）
  + SQLiteKVStore（docstore/index_store，自研）
```

- **向量库**：qdrant 官方服务端（Docker 容器 `llamaindex-qdrant`，程序自动拉起，只绑定 127.0.0.1:16333）。大库启动秒级、常驻内存约 250MB
- **Embedding**：本地 `models/bge-m3`（1024 维），启动时可改选其他模型
- **LLM**：DeepSeek（默认）/ 通义千问 / 智谱 GLM / Ollama / 自定义 OpenAI 兼容，Key 只走环境变量

## 快速部署（新机器）

前置软件：Python 3.11、Git、**Docker Desktop**（需先启动一次；报虚拟化错误则管理员执行 `wsl --install --no-distribution` 后重启）。

```powershell
git clone <仓库地址> LlamaIndex
cd LlamaIndex
python -m pip install uv
uv venv .venv --python 3.11

# torch 按显卡选一行：RTX50 系用 cu128；RTX20/30/40 系用 cu121 的 torch==2.5.1；无 N 卡直接 torch==2.5.1
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt
.venv\Scripts\python.exe download_models.py --model bge-m3   # 约 4.25GB
```

启动：

```powershell
$env:DEEPSEEK_API_KEY="sk-你的key"
$env:LLM_PROVIDER="deepseek"
.venv\Scripts\python.exe run_vector_demo.py        # CLI
# 或
.venv\Scripts\streamlit.exe run app.py --server.port 8501   # Web UI
```

首次启动程序会自动创建 qdrant 容器（拉取镜像约 100MB），看到 `[Qdrant] 服务端就绪` 即成功。

迁移旧机器的已有数据（不重新 embedding）：拷贝 `storage/<库id>/` 和 `kb_configs/` 到新机器，执行 `.venv\Scripts\python.exe migrate_qdrant_to_server.py`。

## 常用命令（CLI）

| 命令 | 说明 |
|---|---|
| `ingest <路径>` | 把文件/文件夹增量加入当前库（自动去重、断点续传） |
| `mode` | 切换检索模式：向量 / 聚合直遍 / 全文搜索 |
| `kbs` | 查看所有知识库状态 |
| `list` | 列出已 ingest 的文件 |
| `rebuild` | 清空当前库重建（无确认） |

## 常用环境变量

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | LLM Key（按所选厂商设置对应变量） |
| `EMBED_MODEL` | 预设 embedding 模型路径，跳过菜单 |
| `QDRANT_MODE` | `server`（默认，Docker 容器）/ `local`（回退内嵌模式，大库启动慢） |
| `QDRANT_HOST` / `QDRANT_PORT` | qdrant 地址（默认 `127.0.0.1:16333`，可指向远程服务器） |

## 文档

- **MANUAL.md** — 完整手册：手动部署逐步指南、命令详解、检索模式、参数配置、常见问题
- `_deploy_pkg/` — 一键部署脚本（deploy.ps1）与部署说明
