# 모델 불러오기
from core.config import config
import gc
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
                gpu_memory_utilization=0.65,
                # max_inference_batch_size=128,
                max_model_len=1024 * 16,
            )
        print("ASR model ready.")
        return self.asr
    
    def __call__(self, *args, **kwds):
        self.asr = self.get_asr()
        return self.asr

    def reset_runtime_caches(self) -> bool:
        if self.asr is None:
            return False

        released = False
        backend_model = getattr(self.asr, "model", None)

        if backend_model is not None:
            reset_prefix_cache = getattr(backend_model, "reset_prefix_cache", None)
            if callable(reset_prefix_cache):
                try:
                    released = bool(reset_prefix_cache()) or released
                except Exception as exc:
                    print(f"[WARN] Failed to reset vLLM prefix cache: {exc}")

            reset_mm_cache = getattr(backend_model, "reset_mm_cache", None)
            if callable(reset_mm_cache):
                try:
                    reset_mm_cache()
                    released = True
                except Exception as exc:
                    print(f"[WARN] Failed to reset vLLM multimodal cache: {exc}")

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception as exc:
            print(f"[WARN] Failed to release CUDA allocator cache: {exc}")

        return released

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


def reset_asr_runtime_caches() -> bool:
    return asrloader.reset_runtime_caches()

aligner_loader = Aligner()
aligner = aligner_loader()

pyannote_pipeline_loader = Diarizer()
pyannote_pipeline = pyannote_pipeline_loader()