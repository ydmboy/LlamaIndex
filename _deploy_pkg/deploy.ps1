# deploy.ps1 — LlamaIndex RAG 一键部署（Windows + PowerShell）
# 用法：在目标机器 git clone 后，于项目根目录执行：
#   powershell -ExecutionPolicy Bypass -File _deploy_pkg\deploy.ps1
$ErrorActionPreference = "Stop"
Write-Host "===== LlamaIndex RAG 部署 =====" -ForegroundColor Cyan

# 1) 检查 Python（需 3.10~3.11，兼容性最佳）
try { $pv = python --version } catch { Write-Host "未检测到 python，请先安装 Python 3.11" -ForegroundColor Red; exit 1 }
Write-Host "Python: $pv"

# 2) 安装 uv（加速依赖安装）
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host "安装 uv ..."
    python -m pip install uv
}

# 3) 创建虚拟环境
if (-not (Test-Path ".venv")) {
    Write-Host "创建虚拟环境 .venv ..."
    uv venv .venv --python 3.11
}

# 4) 先装 torch GPU 版（cu121），避免被 PyPI CPU 版抢占
Write-Host "安装 torch GPU 版 (cu121) ..."
uv pip install torch==2.5.1 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121

# 5) 安装其余依赖
Write-Host "安装其余依赖 ..."
uv pip install -r requirements.txt

# 6) 验证 GPU
Write-Host "验证 torch + CUDA ..."
& .venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

# 7) 下载 embedding 模型（bge-m3，约 4.25 GB）
$modelDir = ".\models\bge-m3"
if (-not (Test-Path $modelDir) -or -not (Test-Path "$modelDir\model.safetensors")) {
    Write-Host ""
    Write-Host "下载 embedding 模型 bge-m3（约 4.25 GB）..." -ForegroundColor Yellow
    & .venv\Scripts\python.exe download_models.py --model bge-m3
} else {
    Write-Host "bge-m3 模型已存在，跳过下载" -ForegroundColor Green
}

Write-Host ""
Write-Host "===== 部署完成 =====" -ForegroundColor Green
Write-Host ""
Write-Host "运行前设置环境变量："
Write-Host '  $env:DEEPSEEK_API_KEY="你的DeepSeek_key"'
Write-Host '  $env:LLM_PROVIDER="deepseek"'
Write-Host '  $env:EMBED_MODEL=".\models\bge-m3"'
Write-Host ""
Write-Host "启动 CLI："
Write-Host "  .venv\Scripts\python.exe run_vector_demo.py"
Write-Host ""
Write-Host "启动 Web UI："
Write-Host '  $env:STREAMLIT_HOME=".\.streamlit"'
Write-Host "  .venv\Scripts\streamlit.exe run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"
Write-Host ""
Write-Host "首次使用：启动后选择数据库 → 用 ingest <路径> 添加文档 → 提问"
