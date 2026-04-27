# 모델 불러오기
from core.config import config
import os

ASR_MODEL_PATH = os.getenv("ASR_MODEL_PATH")
ALIGNER_MODEL_PATH = os.getenv("ALIGNER_MODEL_PATH")
DIARIZE_MODEL_PATH = os.getenv("DIARIZE_MODEL_PATH")
HF_TOKEN = os.getenv("HF_TOKEN")

class ASRModel:
    def __init__(self):
        self.asr = None
        

    def get_asr(self):
        ''' 전사모델을 불러오는 함수'''
        if self.asr is None:
            # 모델 초기화
            from qwen_asr import Qwen3ASRModel
            print("Loading ASR model …")
            self.asr = Qwen3ASRModel.LLM(
                model=ASR_MODEL_PATH,
                gpu_memory_utilization=0.55,
                max_new_tokens=32,
                max_model_len=4096,
            )
        print("ASR model ready.")
        return self.asr
    
    def __call__(self, *args, **kwds):
        self.asr = self.get_asr()
        return self.asr

class Aligner:
    def __init__(self):
        self.aligner = None

    def get_aligner(self):
        ''' 정렬모델을 불러오는 함수 '''
        if self.aligner is None:
            import torch
            from qwen_asr import Qwen3ForcedAligner
            print("Loading ForcedAligner …")
            self.aligner = Qwen3ForcedAligner.from_pretrained(
                ALIGNER_MODEL_PATH,
                dtype=torch.bfloat16,
                device_map="cuda:0",
            )
            print("ForcedAligner ready.")
        return self.aligner
    
    def __call__(self, *args, **kwds):
        self.aligner = self.get_aligner()
        return self.aligner

class Diarizer:
    def __init__(self):
        self.diarizer = None

    def get_pyannote(self):
        ''' pyannote 화자분리 파이프라인을 불러오는 함수'''
        if self.diarizer is None:
            import torch
            from pyannote.audio import Pipeline
            print("Loading pyannote pipeline ...")
            self.diarizer = Pipeline.from_pretrained(
                DIARIZE_MODEL_PATH,
                token=HF_TOKEN or None,
            )
            self.diarizer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print("pyannote pipeline ready.")
        return self.diarizer
    
    def __call__(self, *args, **kwds):
        self.diarizer = self.get_pyannote()
        return self.diarizer

asrloader = ASRModel()
asr = asrloader()

aligner_loader = Aligner()
aligner = aligner_loader()

pyannote_pipeline_loader = Diarizer()
pyannote_pipeline = pyannote_pipeline_loader()