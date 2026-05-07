from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.test_offline_router import router as test_router
from db.mongodb import close_mongodb, init_mongodb


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_mongodb()
    yield
    await close_mongodb()


app = FastAPI(title="Offline Diarization Test API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(test_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}