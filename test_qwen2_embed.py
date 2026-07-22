"""小样本冒烟测试：验证 gte-Qwen2-7B-instruct 能在 RTX 4060 Ti 上加载并产生正确维度向量"""
import os
import sys
import time

# 设置环境（与 run_vector_demo.py 保持一致）
os.environ.setdefault("EMBED_MODEL", "Alibaba-NLP/gte-Qwen2-7B-instruct")

print("=" * 70)
print("Step 1: 检查环境")
print("=" * 70)
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

print()
print("=" * 70)
print("Step 2: 检查 transformers 和 sentence-transformers 版本")
print("=" * 70)
try:
    import transformers
    print(f"transformers: {transformers.__version__} (需要 >=4.39.2)")
except ImportError as e:
    print(f"❌ transformers 未安装: {e}")
    sys.exit(1)

try:
    import sentence_transformers
    print(f"sentence_transformers: {sentence_transformers.__version__}")
except ImportError as e:
    print(f"❌ sentence_transformers 未安装: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("Step 3: 加载 gte-Qwen2-7B-instruct（首次会下载约 14GB，请耐心等待）")
print("=" * 70)
print("开始时间:", time.strftime("%H:%M:%S"))
start = time.time()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)

try:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
        trust_remote_code=True,
        query_instruction=QUERY_INSTRUCTION,
        text_instruction=None,
        max_length=8192,
        embed_batch_size=8,
        model_kwargs={"torch_dtype": torch.float16},
        pooling="last",
    )
    elapsed = time.time() - start
    print(f"✅ 模型加载成功，耗时: {elapsed:.1f} 秒")
except Exception as e:
    print(f"❌ 模型加载失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("Step 4: 检查显存占用")
print("=" * 70)
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"已分配: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"已预留: {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)")
    if allocated / total > 0.95:
        print("⚠️ 警告：显存占用超过 95%，全量构建时可能 OOM")

print()
print("=" * 70)
print("Step 5: 测试 query embedding（验证 query_instruction 自动应用）")
print("=" * 70)
test_query = "湖北有哪些新闻"
print(f"测试 query: {test_query}")
start = time.time()
try:
    q_emb = Settings.embed_model.get_query_embedding(test_query)
    elapsed = time.time() - start
    print(f"✅ query embedding 成功，耗时: {elapsed*1000:.1f} ms")
    print(f"   向量维度: {len(q_emb)} (期望 3584)")
    assert len(q_emb) == 3584, f"维度不对: 期望 3584, 实际 {len(q_emb)}"
except Exception as e:
    print(f"❌ query embedding 失败: {type(e).__name__}: {e}")

print()
print("=" * 70)
print("Step 6: 测试 text embedding（文档端，不带 instruction）")
print("=" * 70)
test_text = "湖北省位于中国中部，是一个新闻丰富的省份。"
print(f"测试 text: {test_text}")
start = time.time()
try:
    t_emb = Settings.embed_model.get_text_embedding(test_text)
    elapsed = time.time() - start
    print(f"✅ text embedding 成功，耗时: {elapsed*1000:.1f} ms")
    print(f"   向量维度: {len(t_emb)} (期望 3584)")
except Exception as e:
    print(f"❌ text embedding 失败: {type(e).__name__}: {e}")

print()
print("=" * 70)
print("Step 7: 测试批量 embedding（模拟构建索引场景）")
print("=" * 70)
test_texts = [
    "湖北省位于中国中部。",
    "湖北日报是省级党报。",
    "武汉是湖北省省会。",
    "长江流经湖北省。",
    "三峡大坝位于湖北宜昌。",
]
print(f"测试批量 ({len(test_texts)} 条):")
for i, t in enumerate(test_texts):
    print(f"  [{i+1}] {t}")
start = time.time()
try:
    embs = Settings.embed_model.get_text_embedding_batch(test_texts)
    elapsed = time.time() - start
    print(f"✅ 批量 embedding 成功，耗时: {elapsed*1000:.1f} ms")
    print(f"   平均每条: {elapsed*1000/len(test_texts):.1f} ms")
    print(f"   推算 1052 文档全量构建 (按每文档 5 块): {elapsed*1000*5260/len(test_texts)/1000/60:.1f} 分钟")
except Exception as e:
    print(f"❌ 批量 embedding 失败: {type(e).__name__}: {e}")

print()
print("=" * 70)
print("Step 8: 验证 query 和 text 在同一表示空间")
print("=" * 70)
try:
    import numpy as np
    q_arr = np.array(q_emb)
    t_arr = np.array(t_emb)
    # 计算余弦相似度
    sim = (q_arr @ t_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(t_arr))
    print(f"query='湖北有哪些新闻' 与 text='湖北省位于中国中部...' 余弦相似度: {sim:.4f}")
    if sim > 0.3:
        print("✅ 相似度正常，query 和 text 在同一空间")
    else:
        print("⚠️ 相似度偏低，可能 query_instruction 未正确应用")
except Exception as e:
    print(f"❌ 相似度计算失败: {e}")

print()
print("=" * 70)
print("测试完成")
print("=" * 70)
print("如果以上全部 ✅，可以执行 rebuild。预计耗时 30-60 分钟。")
print("构建命令：在独立 PowerShell 中运行 python run_vector_demo.py 然后输入 rebuild")
