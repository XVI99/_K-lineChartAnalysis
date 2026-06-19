# -*- coding: utf-8 -*-
"""
cache_manager.py — Redis 缓存管理器

所有数据获取都经由该管理器，避免重复网络请求。
使用 Redis 作为存储后端，支持多进程共享和持久化。
"""

import json
import time
from typing import Optional
import pandas as pd
import redis


class CacheManager:
    """Redis 缓存管理器"""

    _instance: Optional["CacheManager"] = None
    _redis_client: Optional[redis.Redis] = None

    def __new__(cls, host: str = "localhost", port: int = 6379, 
                db: int = 0, expire_seconds: int = 3600):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._host = host
            cls._instance._port = port
            cls._instance._db = db
            cls._instance._expire = expire_seconds
        return cls._instance

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, expire_seconds: int = 3600):
        if self._initialized and self._redis_client is not None:
            return
        self._expire = expire_seconds
        self._host = host
        self._port = port
        self._db = db
        self._connect()
        self._initialized = True

    def _connect(self):
        """建立 Redis 连接"""
        try:
            self._redis_client = redis.Redis(
                host=str(self._host),
                port=int(self._port),
                db=int(self._db),
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            self._redis_client.ping()
            print(f"[Cache] Redis 连接成功: {self._host}:{self._port}/{self._db}")
        except (redis.ConnectionError, redis.TimeoutError, TypeError) as e:
            print(f"[Cache] Redis 连接失败: {e}，将使用无缓存模式")
            self._redis_client = None

    def _serialize(self, df: pd.DataFrame) -> bytes:
        """DataFrame 序列化为 JSON 字节"""
        return df.to_json(date_format="iso").encode("utf-8")

    def _deserialize(self, data: bytes) -> pd.DataFrame:
        """反序列化为 DataFrame"""
        if data is None:
            return None
        from io import StringIO
        df = pd.read_json(StringIO(data.decode("utf-8")), dtype=False)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif df.index.name == "date":
            df.index = pd.to_datetime(df.index)
        return df

    def _is_stale(self, key: str) -> bool:
        """检查 key 是否过期"""
        if self._redis_client is None:
            return True
        ttl = self._redis_client.ttl(key)
        return ttl == -2 or ttl == -1

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """从 Redis 获取数据"""
        if self._redis_client is None:
            return None
        try:
            data = self._redis_client.get(key)
            if data:
                return self._deserialize(data)
            return None
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"[Cache] Redis 获取失败: {e}")
            return None

    def put(self, key: str, data: pd.DataFrame) -> None:
        """存入 Redis，设置过期时间"""
        if self._redis_client is None:
            return
        try:
            serialized = self._serialize(data)
            self._redis_client.setex(key, self._expire, serialized)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"[Cache] Redis 写入失败: {e}")

    def invalidate(self, key: str) -> None:
        """删除指定 key"""
        if self._redis_client is None:
            return
        try:
            self._redis_client.delete(key)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"[Cache] Redis 删除失败: {e}")

    def clear(self) -> None:
        """清除所有缓存 (谨慎使用)"""
        if self._redis_client is None:
            return
        try:
            keys = self._redis_client.keys("*")
            if keys:
                self._redis_client.delete(*keys)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"[Cache] Redis 清除失败: {e}")

    def clear_pattern(self, pattern: str) -> int:
        """按模式清除缓存，返回删除数量"""
        if self._redis_client is None:
            return 0
        try:
            keys = self._redis_client.keys(pattern)
            if keys:
                return self._redis_client.delete(*keys)
            return 0
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"[Cache] Redis 模式清除失败: {e}")
            return 0

    def set_with_days(self, key: str, data: pd.DataFrame, expire_days: int = 30) -> None:
        """存入 Redis，设置按天过期"""
        if self._redis_client is None:
            return
        try:
            serialized = self._serialize(data)
            self._redis_client.setex(key, expire_days * 86400, serialized)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"[Cache] Redis 按天过期写入失败: {e}")

    def exists(self, key: str) -> bool:
        """检查 key 是否存在"""
        if self._redis_client is None:
            return False
        try:
            return bool(self._redis_client.exists(key))
        except (redis.ConnectionError, redis.TimeoutError):
            return False

    @property
    def size(self) -> int:
        """获取缓存键数量"""
        if self._redis_client is None:
            return 0
        try:
            return len(self._redis_client.keys("*"))
        except (redis.ConnectionError, redis.TimeoutError):
            return 0

    @property
    def info(self) -> dict:
        """获取 Redis 状态信息"""
        if self._redis_client is None:
            return {"status": "disconnected"}
        try:
            info = self._redis_client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": self.size,
                "expire_seconds": self._expire,
            }
        except (redis.ConnectionError, redis.TimeoutError) as e:
            return {"status": "error", "error": str(e)}
