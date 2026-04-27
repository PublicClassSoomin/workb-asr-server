# 오디오 전처리 함수
import io
import subprocess
import tempfile
import numpy as np
import soundfile as sf

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