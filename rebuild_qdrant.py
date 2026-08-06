"""一次性脚本：从 docstore 文档重新切块 + 重新 embedding，重建 qdrant collection。

⚠️ 历史脚本（2026-07 一次性去重清理用，已执行完毕）。2026-08-06 起向量库
已切换到 qdrant server 模式（collection 在服务端，不在 storage/<kb>/qdrant/），
本脚本的 local 路径假设已不适用于当前架构，仅存档备查，勿直接运行。

背景：
- storage/newspaper/qdrant 有 147 万个点（19GB），其中约 60 万个是历史重复
  ingest 的残留（docstore 去重后只剩 6984 篇文档，按当前 512/50 切块约 89 万片段）。
- Qdrant local 模式打开时把所有点全量载入内存，是内存膨胀到 97-98% 的主因。
- docstore 节点 JSON 里 embedding 字段为空，且 chunk 节点只存在于 qdrant
  payload 中（docstore 只存文档级去重记录），所以必须重新切块 + 重新 embedding。

切块/embedding 配置与 run_vector_demo 完全一致：
SentenceSplitter(512, 50) + bge-m3 FP16 batch=64（_build_embed_model 小模型分支）。

本脚本不碰旧目录：新 collection 建在 qdrant_new/，验证通过后再替换旧目录。

用法：.venv/Scripts/python.exe rebuild_qdrant.py
"""

import json
import sqlite3
import time

import torch
import qdrant_client
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore.utils import json_to_doc
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

STORAGE_DIR = "storage/newspaper"
DB_PATH = f"{STORAGE_DIR}/docstore.db"
NEW_QDRANT_DIR = f"{STORAGE_DIR}/qdrant_new"
COLLECTION = "default"
EMBED_MODEL_PATH = "models/bge-m3"
CHUNK_SIZE = 512       # retrieval_config.yaml chunk.chunk_size
CHUNK_OVERLAP = 50     # retrieval_config.yaml chunk.chunk_overlap
TOTAL_DOCS = 6984      # 仅用于进度估算
BATCH_SIZE = 512       # 多少片段攒一批做 embedding + 写入


def iter_documents():
    """只读流式遍历 docstore 文档，避免 get_all 一次性物化全表。"""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT value FROM kvstore WHERE collection = 'docstore/data'"
        )
        for (value,) in cur:
            yield json_to_doc(json.loads(value))
    finally:
        conn.close()


def main() -> None:
    # 与 run_vector_demo._build_embed_model 小模型分支保持一致
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_PATH,
        embed_batch_size=64,
        model_kwargs={"torch_dtype": torch.float16},
    )
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    client = qdrant_client.QdrantClient(path=NEW_QDRANT_DIR)
    vs = QdrantVectorStore(client=client, collection_name=COLLECTION)

    def flush(batch):
        # 与 IngestionPipeline 的 embedding 步骤一致：按 EMBED 元数据模式取内容
        embeddings = embed_model.get_text_embedding_batch(
            [n.get_content(metadata_mode=MetadataMode.EMBED) for n in batch]
        )
        for n, emb in zip(batch, embeddings):
            n.embedding = emb
        vs.add(batch)

    batch, total_chunks, total_docs = [], 0, 0
    t0 = time.time()
    for doc in iter_documents():
        total_docs += 1
        for node in splitter.get_nodes_from_documents([doc]):
            if not node.get_content().strip():
                continue
            batch.append(node)
            if len(batch) >= BATCH_SIZE:
                flush(batch)
                total_chunks += len(batch)
                batch.clear()
                elapsed = time.time() - t0
                eta = elapsed / total_docs * (TOTAL_DOCS - total_docs) / 60
                print(
                    f"docs={total_docs}/{TOTAL_DOCS} chunks={total_chunks} "
                    f"elapsed={elapsed / 60:.1f}min eta={eta:.1f}min",
                    flush=True,
                )
    if batch:
        flush(batch)
        total_chunks += len(batch)

    info = client.get_collection(COLLECTION)
    print(
        f"done: docs={total_docs} chunks={total_chunks} "
        f"points_count={info.points_count} elapsed={(time.time() - t0) / 60:.1f}min"
    )
    client.close()


if __name__ == "__main__":
    main()
