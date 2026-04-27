# 몽고db연결
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from core.config import config
from datetime import datetime, timezone, timedelta

client: AsyncIOMotorClient | None = None
db: AsyncIOMotorDatabase | None = None


async def init_mongodb() -> None:
    """애플리케이션 시작 시 MongoDB 클라이언트 초기화"""
    global client, db
    client = AsyncIOMotorClient(config.MONGO_URI)
    db = client[config.MONGO_DB]
    await client.admin.command("ping")


async def close_mongodb() -> None:
    """애플리케이션 종료 시 MongoDB 연결 해제"""
    global client, db
    if client is not None:
        client.close()
        client = None
        db = None


def get_db() -> AsyncIOMotorDatabase:
    if db is None:
        raise RuntimeError("MongoDB가 초기화되지 않았습니다.")
    return db

def upload_mongodb(minutes, meeting_id, workspace_id, duration, meeting_start_time):
    """회의록을 몽고DB에 저장하는 함수"""
    now = datetime.now(timezone.utc) + timedelta(hours=9)
    collection = get_db()["utterances"]
    document = {
        "meeting_id": meeting_id,
        "created_at": now,
        "updated_at": now,
        "utterances": minutes,
        "workspace_id": workspace_id,
        "total_duration_sec": round(duration,0),
        "meeting_start_time": meeting_start_time
    }
    collection.insert_one(document)