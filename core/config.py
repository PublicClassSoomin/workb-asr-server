# 환경상수
from dotenv import load_dotenv
import os

class Config:
    def __init__(self):
        load_dotenv()  # .env 파일에서 환경변수 로드
        self.ASR_MODEL_PATH = os.getenv("ASR_MODEL_PATH")
        self.ALIGNER_MODEL_PATH = os.getenv("ALIGNER_MODEL_PATH")
        self.DIARIZE_MODEL_PATH = os.getenv("DIARIZE_MODEL_PATH")
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.OVERLAP_SEC = float(os.getenv("OVERLAP_SEC", "5.0"))  # 오버랩 길이 (초)
        self.WINDOW_SEC = float(os.getenv("WINDOW_SEC", "30.0"))  # ASR 모델이 한 번에 처리하는 오디오 길이 (초)
        self.MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
        self.MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
        self.MYSQL_USER = os.getenv("MYSQL_USER", "root")
        self.MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
        self.MYSQL_DB = os.getenv("MYSQL_DB", "meeting_assistant")

        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))

        self.REDIS_TTL_SEC=86400

        self.MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.MONGO_DB = os.getenv("MONGO_DB", "meeting_assistant")

        self._PAUSE_THRESHOLD = 1.0  # 문장 병합 시 묵음 간격 임계값 (초)


config = Config()
