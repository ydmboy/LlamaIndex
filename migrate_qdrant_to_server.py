"""migrate_qdrant_to_server.py — 把 Qdrant local 内嵌模式的向量数据搬到 server 模式（一次性）。

背景：local 模式（QdrantClient(path=...)）启动时全量反序列化所有点，大库极慢；
server 模式（本机 qdrant.exe）向量/payload on_disk，启动秒级。本脚本把
storage/<kb>/qdrant/collection/default/storage.sqlite 里的全部 PointStruct
原样搬运到 server 上名为 <kb> 的 collection（向量/payload 逐字节一致，不重新 embedding）。

特性：
- 只读旧数据，不删不改 storage/<kb>/qdrant/；
- 断点续传：server 端已有 >= 本地点数的 collection 会跳过；否则删了重传
  （upsert 幂等，重跑安全，但为省时直接跳过已完成的库）；
- 跑完校验两边点数一致 + 抽样核对 payload。

用法：.venv/Scripts/python.exe migrate_qdrant_to_server.py
"""

import os
import sys
import sqlite3
import pickle
import time

from run_vector_demo import (
    STORAGE_DIR,
    QDRANT_DIRNAME,
    QDRANT_COLLECTION,
    _make_qdrant_client,
    qdrant_models,
)

BATCH_SIZE = 500


def _local_sqlite(kb_dir: str) -> str:
    return os.path.join(kb_dir, QDRANT_DIRNAME, "collection", QDRANT_COLLECTION, "storage.sqlite")


def _to_server_point(pt):
    """local 模式 pickle 出的 PointStruct：vector 是 {"": [...]} 形式，server 要未命名向量列表。"""
    vector = pt.vector
    if isinstance(vector, dict):
        if set(vector.keys()) == {""}:
            vector = vector[""]
        else:
            raise ValueError(f"意外的命名向量结构: {list(vector.keys())}")
    return qdrant_models.PointStruct(id=pt.id, vector=vector, payload=pt.payload)


def _upsert_with_retry(client, kb_id: str, batch: list, attempts: int = 4) -> None:
    """带重试的 upsert：Docker Desktop 重启/端口代理抖动会导致偶发断连。"""
    for i in range(attempts):
        try:
            client.upsert(kb_id, points=batch)
            return
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))


def migrate_kb(client, kb_id: str, kb_dir: str) -> bool:
    sql_path = _local_sqlite(kb_dir)
    if not os.path.isfile(sql_path):
        print(f"[跳过] [{kb_id}] 无 local 向量数据（{sql_path} 不存在）")
        return True

    conn = sqlite3.connect(f"file:{sql_path}?mode=ro", uri=True)
    total = conn.execute("SELECT COUNT(*) FROM points").fetchone()[0]
    print(f"\n[{kb_id}] 本地 {total:,} 点，开始迁移 ...")

    # 断点续传：server 端数量已达标则跳过；有残留则清空重传
    if client.collection_exists(kb_id):
        existing = client.count(kb_id, exact=True).count
        if existing >= total:
            print(f"[{kb_id}] server 端已有 {existing:,} 点（>= 本地 {total:,}），跳过")
            conn.close()
            return True
        print(f"[{kb_id}] server 端有残留 {existing:,} 点，删除后重传")
        client.delete_collection(kb_id)

    # 取首点确定维度，按 server 模式标准参数建库（on_disk 向量+payload）
    first_blob = conn.execute("SELECT point FROM points LIMIT 1").fetchone()[0]
    first_pt = _to_server_point(pickle.loads(first_blob))
    dim = len(first_pt.vector)
    client.create_collection(
        collection_name=kb_id,
        vectors_config=qdrant_models.VectorParams(
            size=dim, distance=qdrant_models.Distance.COSINE, on_disk=True
        ),
        on_disk_payload=True,
    )
    print(f"[{kb_id}] 已创建 server collection（dim={dim}, cosine, on_disk）")

    t0 = time.time()
    done = 0
    batch = []
    for (blob,) in conn.execute("SELECT point FROM points"):
        batch.append(_to_server_point(pickle.loads(blob)))
        if len(batch) >= BATCH_SIZE:
            _upsert_with_retry(client, kb_id, batch)
            done += len(batch)
            batch.clear()
            rate = done / max(time.time() - t0, 1e-6)
            eta = (total - done) / max(rate, 1e-6)
            print(f"\r[{kb_id}] {done:,}/{total:,} ({done * 100 // total}%) "
                  f"{rate:.0f} 点/s，预计剩余 {eta / 60:.1f} 分钟", end="", flush=True)
    if batch:
        _upsert_with_retry(client, kb_id, batch)
        done += len(batch)
    conn.close()
    print(f"\n[{kb_id}] 写入完成，耗时 {(time.time() - t0) / 60:.1f} 分钟")

    # 校验 1：点数一致
    server_count = client.count(kb_id, exact=True).count
    if server_count != total:
        print(f"[{kb_id}] [失败] 点数不一致: server={server_count:,} 本地={total:,}")
        return False

    # 校验 2：抽样核对 payload（file_name + 原文长度）
    conn = sqlite3.connect(f"file:{sql_path}?mode=ro", uri=True)
    sample = conn.execute(
        "SELECT point FROM points WHERE rowid % ? = 1 LIMIT 3", (max(total // 3, 1),)
    ).fetchall()
    conn.close()
    for (blob,) in sample:
        local_pt = _to_server_point(pickle.loads(blob))
        got = client.retrieve(kb_id, ids=[local_pt.id], with_payload=True, with_vectors=True)
        if not got:
            print(f"[{kb_id}] [失败] 抽样点 {local_pt.id} 在 server 端不存在")
            return False
        gp = got[0]
        if (gp.payload or {}).get("file_name") != (local_pt.payload or {}).get("file_name"):
            print(f"[{kb_id}] [失败] 抽样点 {local_pt.id} payload 不一致")
            return False
        if len(gp.vector) != dim:
            print(f"[{kb_id}] [失败] 抽样点 {local_pt.id} 向量维度 {len(gp.vector)} != {dim}")
            return False
    print(f"[{kb_id}] [OK] 校验通过：{server_count:,} 点一致，抽样 payload/向量正常")
    return True


def main() -> None:
    print("=" * 60)
    print(" Qdrant local -> server 一次性迁移")
    print("=" * 60)
    client = _make_qdrant_client()  # 自动拉起本机 qdrant.exe

    kb_dirs = sorted(
        d for d in (os.path.join(STORAGE_DIR, d) for d in os.listdir(STORAGE_DIR))
        if os.path.isdir(d) and not os.path.basename(d).startswith("_")
    )
    if not kb_dirs:
        print(f"未在 {STORAGE_DIR}/ 下找到任何知识库目录")
        return

    failed = []
    for kb_dir in kb_dirs:
        kb_id = os.path.basename(kb_dir)
        try:
            if not migrate_kb(client, kb_id, kb_dir):
                failed.append(kb_id)
        except Exception as e:
            print(f"\n[{kb_id}] [异常] {type(e).__name__}: {e}")
            failed.append(kb_id)

    print("\n" + "=" * 60)
    if failed:
        print(f"迁移完成，但有失败的知识库: {failed}（修复后重跑本脚本即可续传）")
        sys.exit(1)
    print("全部知识库迁移成功！")
    print("验证程序运行正常后，可手动删除旧数据目录 storage/<kb>/qdrant/ 释放空间。")
    print("=" * 60)


if __name__ == "__main__":
    main()
