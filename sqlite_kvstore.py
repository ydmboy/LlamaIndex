"""SQLite 后端 KVStore：LlamaIndex BaseKVStore 的 SQLite 实现。

背景：官方没有 llama-index-storage-kvstore-sqlite 包，这里自行实现。
用途：作为 KVDocumentStore / KVIndexStore 的底层存储，替代 SimpleKVStore 的
"启动全量加载 JSON + 每次落盘全量重写"——改为按需查询、增量写入：

- 哈希去重检查从全表加载变为 O(1) SQL 查询；
- 写盘增量提交，不再随库规模线性变慢；
- 原子性由 SQLite 事务 + WAL 保证，写盘途中断电/强杀不会留下截断文件
  （此前 SimpleKVStore 时代的 tmp+replace 补丁随之退役）。

写策略：put/put_all 立即执行 SQL 但延迟 commit，攒够 _COMMIT_EVERY 条或显式
commit() 时落盘——ingest 的批次检查点处显式 commit，崩溃最多丢失上一检查点
之后的数据（与原有"批次落盘"语义一致）。get/get_all 走同一连接，
未 commit 的数据对读操作即时可见。
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional

from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore

_COMMIT_EVERY = 1000  # 每积累多少条未提交写自动 commit 一次


class SQLiteKVStore(BaseKVStore):
    """BaseKVStore 的 SQLite 实现。表结构：(collection, key) 联合主键，value 为 JSON。"""

    def __init__(self, db_path: str):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kvstore ("
            "collection TEXT NOT NULL, "
            "key TEXT NOT NULL, "
            "value TEXT NOT NULL, "
            "PRIMARY KEY (collection, key))"
        )
        self._conn.commit()
        self._pending = 0

    # ---- 内部 ----
    def _bump(self, n: int = 1) -> None:
        self._pending += n
        if self._pending >= _COMMIT_EVERY:
            self.commit()

    def commit(self) -> None:
        """显式提交（批次检查点 / 程序退出前调用）。"""
        with self._lock:
            if self._pending:
                self._conn.commit()
                self._pending = 0

    def close(self) -> None:
        self.commit()
        self._conn.close()

    # ---- BaseKVStore 接口 ----
    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kvstore (collection, key, value) VALUES (?, ?, ?)",
                (collection, key, json.dumps(val, ensure_ascii=False)),
            )
            self._bump()

    async def aput(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        self.put(key, val, collection)

    def put_all(self, kv_pairs, collection: str = DEFAULT_COLLECTION, batch_size: int = 100) -> None:
        """批量写入（kv_pairs 为 (key, val) 元组列表）：一次 executemany，
        覆盖基类"batch_size 必须为 1 否则报错"的逐条 put 循环。"""
        rows = [
            (collection, key, json.dumps(val, ensure_ascii=False))
            for key, val in kv_pairs
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kvstore (collection, key, value) VALUES (?, ?, ?)",
                rows,
            )
            self._bump(len(rows))

    async def aput_all(self, kv_pairs, collection: str = DEFAULT_COLLECTION, batch_size: int = 100) -> None:
        self.put_all(kv_pairs, collection, batch_size)

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kvstore WHERE collection = ? AND key = ?",
                (collection, key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    async def aget(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        return self.get(key, collection)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM kvstore WHERE collection = ?", (collection,)
            ).fetchall()
        return {key: json.loads(value) for key, value in rows}

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        return self.get_all(collection)

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM kvstore WHERE collection = ? AND key = ?",
                (collection, key),
            )
            self._bump()
            return cur.rowcount > 0

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        return self.delete(key, collection)
