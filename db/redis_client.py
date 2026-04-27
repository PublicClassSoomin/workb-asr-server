# 레디스 연결
import redis.asyncio as aioredis
from core.config import config

redis: aioredis.Redis | None = None


async def init_redis() -> None:
    """애플리케이션 시작 시 Redis 클라이언트 초기화"""
    global redis
    redis = aioredis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        db=config.REDIS_DB,
        decode_responses=True,
    )
    await redis.ping()


async def close_redis() -> None:
    """애플리케이션 종료 시 Redis 연결 해제"""
    global redis
    if redis is not None:
        await redis.aclose()
        redis = None


def get_redis() -> aioredis.Redis:
    if redis is None:
        raise RuntimeError("Redis가 초기화되지 않았습니다.")
    return redis