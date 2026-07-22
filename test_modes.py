"""
测试三种检索模式的切换和效果。
只测检索逻辑（不测 REPL 交互），快速验证三种模式都能正常工作。

用法：
    .venv\Scripts\python.exe test_modes.py
"""
import os
import sys
import time
import torch
from pathlib import Path

# 复用主项目的模块
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 导入主项目的函数
from run_vector_demo import (
    _handle_aggregate_query,
    _handle_fulltext_query,
    _get_fulltext_searcher,
    RETRIEVAL_MODES,
)

STORAGE_DIR = "./storage/beijing_daily"
INDEX_ID = "vector_index"
EMBED_MODEL_PATH = r"C:\code\LlamaIndex\models\bge-m3"
CHUNK_SIZE = 512


def main():
    # 1) 加载 embedding + 索引
    print("[1/4] 加载 embedding 模型 ...")
    t0 = time.time()
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_PATH,
        embed_batch_size=64,
        model_kwargs={"torch_dtype": torch.float16},
    )
    Settings.chunk_size = CHUNK_SIZE
    print(f"    -> 完成，耗时 {time.time()-t0:.1f}s")

    print("\n[2/4] 加载索引 ...")
    t0 = time.time()
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context, index_id=INDEX_ID)
    print(f"    -> 完成，耗时 {time.time()-t0:.1f}s")

    # 2) 测试模式1：向量检索（用 retriever 模拟，不调 LLM）
    print("\n[3/4] 测试模式1：向量检索（retriever 模拟）")
    print("-" * 60)
    query = "人工智能"
    retriever = index.as_retriever(similarity_top_k=5)
    t0 = time.time()
    nodes = retriever.retrieve(query)
    elapsed = time.time() - t0
    print(f"  查询: 「{query}」 | top_k=5 | 耗时 {elapsed:.2f}s")
    for i, node in enumerate(nodes[:3], 1):
        score = node.score if node.score else 0
        preview = node.node.get_content()[:100].replace("\n", " ")
        print(f"    {i}. score={score:.4f} | {preview}...")

    # 3) 测试模式2：聚合直遍
    print("\n[4/4] 测试模式2：聚合直遍")
    print("-" * 60)
    query = "列出所有文章标题"
    print(f"  查询: 「{query}」")
    _handle_aggregate_query(query, index)

    # 4) 测试模式3：全文搜索
    print("\n[附加] 测试模式3：全文搜索（BM25）")
    print("-" * 60)
    query = "人工智能"
    print(f"  查询: 「{query}」")
    _handle_fulltext_query(query, index, top_k=5)

    # 5) 再测一个全文搜索查询（验证索引已缓存，第二次更快）
    print("\n[附加] 全文搜索第二次查询（验证索引缓存）")
    print("-" * 60)
    query = "地铁项目"
    print(f"  查询: 「{query}」")
    searcher = _get_fulltext_searcher(index)
    print(f"  索引状态: {searcher.status}")
    _handle_fulltext_query(query, index, top_k=5)

    # 总结
    print("\n" + "=" * 60)
    print("三种检索模式测试总结：")
    print("=" * 60)
    print(f"  1. 向量检索    - 语义匹配，需 LLM 生成回答，top-k 返回")
    print(f"  2. 聚合直遍    - 正则匹配，100% 覆盖，适合列举类查询")
    print(f"  3. 全文搜索    - BM25 排序，关键词精确匹配，不调 LLM")
    print(f"  索引状态: {searcher.status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
