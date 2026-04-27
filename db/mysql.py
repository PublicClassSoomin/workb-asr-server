# mysql 연결
import aiomysql
import asyncio
import json
from core.config import config

pool: aiomysql.Pool | None = None


async def init_mysql_pool() -> None:
    """애플리케이션 시작 시 커넥션 풀 초기화"""
    global pool
    pool = await aiomysql.create_pool(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        db=config.MYSQL_DB,
        charset="utf8mb4",
        autocommit=False,
        minsize=1,
        maxsize=10,
    )


async def close_mysql_pool() -> None:
    """애플리케이션 종료 시 커넥션 풀 해제"""
    global pool
    if pool is not None:
        pool.close()
        await pool.wait_closed()
        pool = None

async def _get_pool() -> aiomysql.Pool:
    """풀이 없거나 끊긴 경우 재생성"""
    global pool
    if pool is None or pool.closed:
        await init_mysql_pool()
    return pool

async def fetch_all(query: str, params: tuple = ()):
    for attempt in range(3):
        try:
            async with (await _get_pool()).acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(query, params)
                    return await cursor.fetchall()
        except aiomysql.OperationalError:
            global pool
            pool = None
            if attempt == 2:
                raise
            await asyncio.sleep(1)


async def fetch_one(query: str, params: tuple = ()):
    for attempt in range(3):
        try:
            async with (await _get_pool()).acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(query, params)
                    return await cursor.fetchone()
        except aiomysql.OperationalError:
            global pool
            pool = None
            if attempt == 2:
                raise
            await asyncio.sleep(1)


async def execute(query: str, params: tuple = ()):
    for attempt in range(3):
        try:
            async with (await _get_pool()).acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(query, params)
                    return cursor.rowcount
        except aiomysql.OperationalError:
            global pool
            pool = None
            if attempt == 2:
                raise
            await asyncio.sleep(1)
        
# 회의 참가자 조회
async def get_participants(meeting_id: str):
    query_id = "SELECT user_id FROM meeting_participants WHERE meeting_id = %s"
    rows_id = await fetch_all(query_id, (meeting_id,))
    ids = [int(row["user_id"]) for row in rows_id]
    query_name = "SELECT name FROM users WHERE id IN %s"
    rows_name = await fetch_all(query_name, (ids,))
    names = [row["name"] for row in rows_name]
    return dict(zip(ids, names))

# 회의 참가자 임베딩 조회
async def get_participants_embeddings(participant_ids: list):
    if not participant_ids:
        return {}
    
    s = ",".join(["%s"] * len(participant_ids))
    query = f"SELECT user_id, voice_embedding FROM speaker_profiles WHERE user_id IN ({s})"
    rows = await fetch_all(query, tuple(participant_ids))
    
    # DB 반환 순서가 보장되지 않으므로 user_id 기준으로 매핑
    emb_by_uid = {row["user_id"]: json.loads(row["voice_embedding"]) for row in rows}
    return [emb_by_uid[uid] for uid in participant_ids if uid in emb_by_uid]

# 회의 정보 조회 (workspace_id 등)
async def get_meeting_info(meeting_id: str):
    query = "SELECT workspace_id FROM meetings WHERE id = %s"
    return await fetch_one(query, (meeting_id,))

