# 오디오 전처리 함수
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import soundfile as sf


FFMPEG_BIN = "ffmpeg"

LOW_CUT_HZ = 100

AFFTDN_NOISE_REDUCTION_DB = 5.0
AFFTDN_NOISE_FLOOR_DB = -40.0
ANLMDN_STRENGTH = 0.00005
ANLMDN_PATCH_MS = 1
ANLMDN_RESEARCH_MS = 12
ANLMDN_SMOOTH = 5

COMPRESSOR_THRESHOLD_DB = -30.0
COMPRESSOR_RATIO = 20.0
COMPRESSOR_ATTACK = 15.0
COMPRESSOR_RELEASE = 200.0
COMPRESSOR_MAKEUP = 5.0

LIMITER_DB = -1.5
LIMITER_ATTACK = 15.0
LIMITER_RELEASE = 200.0

def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        # 이미 목표 샘플레이트 → 불필요한 연산 없이 반환
        return wav.astype(np.float32, copy=False)
    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)                        # 총 재생 시간(초)
    n16 = int(round(dur * 16000))                         # 16kHz 기준 목표 샘플 수
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)  # 원본 시간 좌표
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)           # 목표 시간 좌표
    return np.interp(x_new, x_old, wav).astype(np.float32)


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)


def build_reference_preprocess_filter() -> str:
    compressor_threshold = db_to_linear(COMPRESSOR_THRESHOLD_DB)
    limiter_limit = db_to_linear(LIMITER_DB)

    return ",".join(
        [
            f"highpass=f={LOW_CUT_HZ}:p=2",
            "afftdn="
            f"nr={AFFTDN_NOISE_REDUCTION_DB}:"
            f"nf={AFFTDN_NOISE_FLOOR_DB}:"
            "tn=1",
            "acompressor="
            f"threshold={compressor_threshold:.6f}:"
            f"ratio={COMPRESSOR_RATIO}:"
            f"attack={COMPRESSOR_ATTACK}:"
            f"release={COMPRESSOR_RELEASE}",
            "acompressor="
            f"threshold={compressor_threshold:.6f}:"
            f"ratio={COMPRESSOR_RATIO}:"
            f"attack={COMPRESSOR_ATTACK}:"
            f"release={COMPRESSOR_RELEASE}:"
            f"makeup={COMPRESSOR_MAKEUP}",
            "alimiter="
            f"limit={limiter_limit:.6f}:"
            "level=true:"
            f"attack={LIMITER_ATTACK}:"
            f"release={LIMITER_RELEASE}",
            "anlmdn="
            f"s={ANLMDN_STRENGTH}:"
            f"p={ANLMDN_PATCH_MS}ms:"
            f"r={ANLMDN_RESEARCH_MS}ms:"
            f"m={ANLMDN_SMOOTH}",
        ]
    )


def load_audio_file_wav16k(audio_path: Path) -> np.ndarray:
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return resample_to_16k(wav, sr)


def preprocess_audio_file(input_path: Path, output_path: Path, audio_filter: str | None = None) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    if input_path.resolve() == output_path.resolve():
        raise ValueError("입력 파일과 출력 파일은 달라야 합니다.")

    if shutil.which(FFMPEG_BIN) is None:
        raise FileNotFoundError(f"ffmpeg 실행 파일을 찾지 못했습니다: {FFMPEG_BIN}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-filter:a",
        audio_filter or build_reference_preprocess_filter(),
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            "ffmpeg 전처리에 실패했습니다.\n"
            f"command: {' '.join(command)}\n"
            f"stderr:\n{stderr}"
        )

    return output_path


def preprocess_saved_audio(input_path: Path, output_path: Path | None = None) -> tuple[Path, np.ndarray]:
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_preprocessed.wav")

    preprocess_audio_file(input_path, output_path)
    return output_path, load_audio_file_wav16k(output_path)


def preprocess_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    wav16k = bytes_to_wav16k(audio_bytes)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.wav"
        output_path = Path(tmpdir) / "output_preprocessed.wav"
        sf.write(str(input_path), wav16k, 16000)
        _, processed_wav = preprocess_saved_audio(input_path, output_path)
        return processed_wav

def bytes_to_wav16k(audio_bytes: bytes, raw_pcm_sr: int = 48000) -> np.ndarray:
    try:
        # 1단계: soundfile로 메모리 내 직접 파싱
        with io.BytesIO(audio_bytes) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
    except Exception:
        # 2단계: libsndfile 미지원 포맷(WebM/Opus 등) → ffmpeg 변환
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name
        tmp_out_path = tmp_in_path + ".wav"
        ffmpeg_ok = False
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_out_path],
                check=True, capture_output=True,
            )
            wav, sr = sf.read(tmp_out_path, dtype="float32", always_2d=False)
            ffmpeg_ok = True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ffmpeg 변환 실패 (exit={e.returncode})")
            print(f"[ERROR] ffmpeg stderr: {e.stderr.decode(errors='replace')}")
        except Exception as e:
            print(f"[ERROR] ffmpeg 예외: {e}")
        finally:
            import os
            for p in (tmp_in_path, tmp_out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if not ffmpeg_ok:
            # 3단계: raw PCM fallback — Int16 → Float32 → Int32 순서로 시도
            wav = None
            if len(audio_bytes) % 2 == 0:
                try:
                    pcm = np.frombuffer(audio_bytes, dtype=np.int16).copy()
                    wav = pcm.astype(np.float32) / 32768.0
                    sr = raw_pcm_sr
                except Exception:
                    pass
            if wav is None and len(audio_bytes) % 4 == 0:
                try:
                    wav = np.frombuffer(audio_bytes, dtype=np.float32).copy()
                    sr = raw_pcm_sr
                except Exception:
                    pass
            if wav is None and len(audio_bytes) % 4 == 0:
                try:
                    pcm = np.frombuffer(audio_bytes, dtype=np.int32).copy()
                    wav = pcm.astype(np.float32) / 2147483648.0
                    sr = raw_pcm_sr
                except Exception:
                    pass
            if wav is None:
                raise ValueError(f"오디오 파싱 실패: 지원하지 않는 포맷 (bytes={len(audio_bytes)})")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        # 스테레오 이상 → 채널 평균으로 모노 다운믹스
        wav = wav.mean(axis=1)
    return resample_to_16k(wav, sr)