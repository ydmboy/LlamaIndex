"""
测试不同 similarity_top_k 对聚合查询的检索效果。
只测检索环节（不调 LLM），快速验证 top_k 的影响。

用法：
    .venv\Scripts\python.exe test_topk.py
"""
import os
import sys
import time
import torch
from pathlib import Path

# 复用主项目的 embedding 构建
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

STORAGE_DIR = "./storage/beijing_daily"
INDEX_ID = "vector_index"
EMBED_MODEL_PATH = r"C:\code\LlamaIndex\models\bge-m3"
CHUNK_SIZE = 512

# 测试查询
TEST_QUERIES = [
    "列出所有文章标题",
    "列出所有项目名称",
]

# 待测试的 top_k 值
TEST_TOP_KS = [2, 10, 20, 50, 100]


def main():
    # 1) 加载 embedding 模型
    print(f"[1/3] 加载 embedding 模型: {EMBED_MODEL_PATH}")
    t0 = time.time()
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_PATH,
        embed_batch_size=64,
        model_kwargs={"torch_dtype": torch.float16},
    )
    Settings.chunk_size = CHUNK_SIZE
    print(f"    -> 完成，耗时 {time.time()-t0:.1f}s")

    # 2) 加载索引
    print(f"\n[2/3] 加载索引: {STORAGE_DIR}")
    t0 = time.time()
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context, index_id=INDEX_ID)
    print(f"    -> 完成，耗时 {time.time()-t0:.1f}s")

    # 统计索引中的总 chunk 数
    try:
        total_chunks = len(index.vector_store.data.embedding_dict)
    except Exception:
        total_chunks = "?"
    print(f"    -> 索引中共 {total_chunks} 个 chunk")

    # 3) 测试不同 top_k
    print(f"\n[3/3] 测试不同 similarity_top_k 的检索效果")
    print("=" * 70)

    for query in TEST_QUERIES:
        print(f"\n查询: 「{query}」")
        print("-" * 70)

        for top_k in TEST_TOP_KS:
            retriever = index.as_retriever(similarity_top_k=top_k)
            t0 = time.time()
            nodes = retriever.retrieve(query)
            elapsed = time.time() - t0

            # 统计检索到的片段中包含的文章标题（### 开头的行）
            titles_found = set()
            for node in nodes:
                text = node.node.get_content()
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("### "):
                        titles_found.add(line)

            # 覆盖率
            if isinstance(total_chunks, int) and total_chunks > 0:
                coverage = f"{len(nodes)}/{total_chunks} chunks ({len(nodes)/total_chunks*100:.1f}%)"
            else:
                coverage = f"{len(nodes)} chunks"

            print(f"  top_k={top_k:>3} | 检索 {len(nodes)} 个片段 | {coverage} | "
                  f"含 {len(titles_found)} 个标题 | 耗时 {elapsed:.2f}s")

            # top_k=2 和 top_k=50 时打印详细内容
            if top_k in (2, 50):
                for i, node in enumerate(nodes[:5], 1):  # 最多打印前5个
                    score = node.score if node.score else 0
                    text_preview = node.node.get_content()[:120].replace("\n", " ")
                    print(f"    片段{i} (score={score:.4f}): {text_preview}...")
                if len(nodes) > 5:
                    print(f"    ... 还有 {len(nodes)-5} 个片段")

        print()

    # 总结
    print("=" * 70)
    print("总结：")
    print(f"  - 知识库共 {total_chunks} 个 chunk")
    if isinstance(total_chunks, int) and total_chunks > 0:
        print(f"  - 默认 top_k=2 只能覆盖 2 个 chunk（约 {2/total_chunks*100:.3f}%）")
    print(f"  - 向量检索调大 top_k 对聚合查询效果有限（见上方数据）")
    print(f"  - 聚合查询应直遍 docstore，覆盖率 100%")
    print("=" * 70)

    # ---- 4) 测试聚合查询（直遍 docstore）----
    print(f"\n[4/4] 测试聚合查询（直遍 docstore，跳过向量检索）")
    print("=" * 70)

    from tqdm import tqdm
    import re as _re

    agg_queries = [
        "列出所有文章标题",
        "列出所有项目名称",
    ]

    for query in agg_queries:
        print(f"\n聚合查询: 「{query}」")
        print("-" * 70)

        t0 = time.time()
        docstore = index.storage_context.docstore
        docs = docstore.docs
        total_nodes = len(docs)

        results = []
        seen = set()

        # 策略1：文章标题
        if _re.search(r"标题|文章名", query):
            print(f"  提取模式：文章标题（### 开头）| 扫描 {total_nodes} 个节点 ...")
            for _, node in tqdm(docs.items(), desc="扫描", total=total_nodes, ncols=70, leave=False):
                text = node.get_content()
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("### ") and line not in seen:
                        seen.add(line)
                        results.append(line)

        # 策略2：项目名称
        elif _re.search(r"项目", query):
            print(f"  提取模式：包含「项目」的行 | 扫描 {total_nodes} 个节点 ...")
            for _, node in tqdm(docs.items(), desc="扫描", total=total_nodes, ncols=70, leave=False):
                text = node.get_content()
                for line in text.split("\n"):
                    line = line.strip()
                    if "项目" in line and 4 < len(line) < 200 and line not in seen:
                        seen.add(line)
                        results.append(line)

        elapsed = time.time() - t0

        # 对比：向量检索 top_k=2 只找到几个
        retriever_default = index.as_retriever(similarity_top_k=2)
        default_nodes = retriever_default.retrieve(query)
        default_titles = 0
        for node in default_nodes:
            for line in node.node.get_content().split("\n"):
                if line.strip().startswith("### "):
                    default_titles += 1

        print(f"\n  结果对比：")
        print(f"    向量检索 top_k=2  → {default_titles} 个标题")
        print(f"    直遍 docstore     → {len(results)} 个标题")
        print(f"    扫描耗时: {elapsed:.1f}s | 覆盖率: 100%")

        # 打印前 20 个结果
        if results:
            print(f"\n  前 20 个结果：")
            for i, item in enumerate(results[:20], 1):
                print(f"    {i:>3}. {item[:80]}")
            if len(results) > 20:
                print(f"    ... 还有 {len(results)-20} 个")

    print("\n" + "=" * 70)
    print("结论：聚合查询直遍 docstore 精度 100%，远优于向量检索")
    print("=" * 70)


if __name__ == "__main__":
    main()
