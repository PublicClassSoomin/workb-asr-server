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
                    await conn.commit()
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
    if not ids:
        return {}
    s = ",".join(["%s"] * len(ids))
    query_name = f"SELECT id, name FROM users WHERE id IN ({s})"
    rows_name = await fetch_all(query_name, tuple(ids))
    return {int(row["id"]): row["name"] for row in rows_name}

# 회의 참가자 임베딩 조회
async def get_participants_embeddings(participant_ids: list):
    if not participant_ids:
        return {}
    
    s = ",".join(["%s"] * len(participant_ids))
    query = f"SELECT user_id, voice_embedding FROM speaker_profiles WHERE user_id IN ({s})"
    rows = await fetch_all(query, tuple(participant_ids))
    
    # DB 반환 순서가 보장되지 않으므로 user_id 기준으로 매핑하여 dict 반환
    # voice_embedding이 NULL이거나 빈 값인 경우 스킵
    emb_by_uid = {}
    for row in rows:
        raw = row["voice_embedding"]
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not parsed:  # 빈 리스트 [] 등 스킵
            continue
        emb_by_uid[row["user_id"]] = parsed
    return emb_by_uid

# 회의 정보 조회 (workspace_id 등)
async def get_meeting_info(meeting_id: str):
    query = "SELECT workspace_id FROM meetings WHERE id = %s"
    return await fetch_one(query, (meeting_id,))

async def save_user_embedding(user_id: int, embedding: str):
    # 기존 임베딩이 있으면 업데이트, 없으면 삽입
    query_check = "SELECT COUNT(*) AS count FROM speaker_profiles WHERE user_id = %s"
    row = await fetch_one(query_check, (user_id,))
    
    if row["count"] > 0:
        query_update = "UPDATE speaker_profiles SET voice_embedding = %s WHERE user_id = %s"
        await execute(query_update, (embedding, user_id))
    else:
        query_insert = (
            "INSERT INTO speaker_profiles (user_id, voice_embedding, workspace_id, is_verified, created_at, diarization_method, updated_at) "
            "VALUES (%s, %s, (SELECT workspace_id FROM users WHERE id = %s), 1, NOW(), %s, NOW())"
        )
        await execute(query_insert, (user_id, embedding, user_id, 'diarization'))
    


