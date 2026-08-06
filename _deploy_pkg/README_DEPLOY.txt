LlamaIndex RAG 系统 — 部署说明
================================

项目文件（git clone 后即有）：
  run_vector_demo.py        主脚本（CLI 交互式 RAG 问答）
  app.py                    Streamlit Web UI
  generate_kb_configs.py    批量生成知识库配置
  download_models.py        下载 embedding 模型
  highlight_config.json     高亮样式配置
  kb_configs/               数据库配置（报纸/代码/通用知识 三个类别库）
  requirements.txt          依赖清单（不含 torch）
  _deploy_pkg/deploy.ps1    一键部署脚本
  _deploy_pkg/requirements.txt  部署用依赖清单

一、部署步骤（目标机器）
  1. git clone <仓库地址> LlamaIndexRAG
  2. cd LlamaIndexRAG
  3. powershell -ExecutionPolicy Bypass -File _deploy_pkg\deploy.ps1
     脚本会自动：
       创建 .venv → 安装 uv → 装 torch GPU 版 → 装其余依赖
       → 验证 CUDA → 下载 bge-m3 模型（约 4.25 GB）→ 检查 docker
  4. 首次部署需联网下载约 7 GB（torch 2.5GB + 模型权重 4.25GB + 依赖包）
     另需安装 Docker Desktop（向量库 qdrant 以容器运行，程序启动时自动拉取镜像）

二、运行
  # 设置环境变量
  $env:DEEPSEEK_API_KEY="你的key"
  $env:LLM_PROVIDER="deepseek"
  $env:EMBED_MODEL=".\models\bge-m3"

  # 启动 CLI
  .venv\Scripts\python.exe run_vector_demo.py

  # 或启动 Web UI
  $env:STREAMLIT_HOME=".\.streamlit"
  .venv\Scripts\streamlit.exe run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

三、架构说明（按类别建库）
  系统有三个独立数据库，各有独立存储目录：
    newspaper  报纸数据库      ./storage/newspaper
    code       代码知识数据库   ./storage/code
    knowledge  通用知识数据库   ./storage/knowledge

  启动后选择一个数据库 → 后续所有 ingest 写入该数据库。
  切换数据库需退出重启。

四、REPL 命令
  任意问题          即时 RAG 问答（根据检索模式分派）
  ingest <路径>     将文件/文件夹加入当前数据库
  embed <路径>      仅向量化查看（不写入索引）
  list              列出当前数据库已 ingest 的文件
  mode              切换检索模式（向量/聚合/全文）
  kbs               查看所有可用数据库
  rebuild           清空当前数据库，重建空索引
  clear/cls         清屏
  exit/quit/退出    退出

五、检索模式
  vector    向量检索（语义匹配，LLM 生成回答）—— 默认
  aggregate 聚合直遍（正则匹配，100%覆盖，适合列举类查询）
  fulltext  全文搜索（倒排索引 + BM25 排序，关键词精确匹配）

六、配置项（环境变量，均可选）
  DEEPSEEK_API_KEY  DeepSeek API Key（必需）
  DEEPSEEK_MODEL    DeepSeek 模型名（默认 deepseek-v4-pro，可选 deepseek-v4-flash）
  LLM_PROVIDER      LLM 提供商（默认 deepseek，可选 qwen/glm/ollama）
  EMBED_MODEL       embedding 模型路径（默认 .\models\bge-m3）
  CHUNK_SIZE        文本分块大小（默认 512）
  CHUNK_OVERLAP     分块重叠（默认 50）

七、注意事项
  - torch 为 GPU 版(cu121)，需 NVIDIA 显卡；无 GPU 改装 CPU 版：
      uv pip install torch==2.5.1
  - 向量库为 qdrant server 模式：程序启动时自动拉起 Docker 容器 llamaindex-qdrant
    （需要 Docker Desktop；只绑定 127.0.0.1:16333，不弹防火墙）；
    数据在 Docker volume llamaindex-qdrant-data。勿改用 qdrant Windows 裸 exe
    （其删除/段合并 rename 有 bug，写入负载下会泄漏段文件耗尽磁盘）
  - 首次启动选择数据库后为空库，需用 ingest 添加文档
  - DEEPSEEK_API_KEY 请勿写入文件，用环境变量传递
  - embedding 模型不在 git 中，由 deploy.ps1 或 download_models.py 下载
