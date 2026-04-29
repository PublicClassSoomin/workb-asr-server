# 메인 애플리케이션
from core.config import config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router
from core.models import asr, aligner, pyannote_pipeline
from db.mysql import init_mysql_pool, close_mysql_pool
from db.redis_client import init_redis, close_redis
from db.mongodb import init_mongodb, close_mongodb
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_mysql_pool()
    await init_redis()
    await init_mongodb()
    yield
    await close_mysql_pool()
    await close_redis()
    await close_mongodb()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "ok"}








