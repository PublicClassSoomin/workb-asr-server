# 화자 분리 함수
import numpy as np
from core.config import config
from core.models import aligner, pyannote_pipeline, asr
from scipy.spatial.distance import cosine
import torch
from datetime import timedelta

MAX_ALIGN_SEC = config.WINDOW_SEC
_PAUSE_THRESHOLD = config._PAUSE_THRESHOLD
_SENT_ENDERS = frozenset("。．.！!？?")   # 구두점 기반 (보조)

def serialize_timestamps(ts_results):
    """ForcedAligner 결과를 JSON 직렬화 가능한 리스트로 변환."""
    if not ts_results or not ts_results[0]:
        return []
    return [
        {
            "text": seg.text,
            "start": round(float(seg.start_time), 3),
            "end": round(float(seg.end_time), 3),
        }
        for seg in ts_results[0]
    ]

def align_chunk(audio_chunks, text, language, time_offset):

    if not audio_chunks or not text.strip():
        return []
    audio = np.concatenate(audio_chunks)
    if audio.shape[0] > MAX_ALIGN_SEC * 16000:
        return []
    try:
        ts_results = aligner.align(
            audio=(audio, 16000),
            text=text.strip(),
            language=language,
        )
        timestamps = serialize_timestamps(ts_results)
        for t in timestamps:
            t["start"] = round(t["start"] + time_offset, 3)
            t["end"] = round(t["end"] + time_offset, 3)
        return timestamps
    except Exception as e:
        print(f"Alignment failed: {e}")
        return []
    
def run_diarization(wav16k: np.ndarray, embedding_speakers) -> tuple[list[dict], dict[str, str]]:
    """
    화자분리를 수행하여 화자 세그먼트 리스트와 화자 이름 맵을 반환.
    Returns: (segments, spk_name_map)
      - segments: [{"speaker": "spk_01", "start": 0.0, "end": 2.5}, ...]
      - spk_name_map: {"spk_01": "김사과", "spk_02": "알 수 없음 Speaker 2", ...}
    """
    waveform = torch.from_numpy(wav16k).unsqueeze(0)  # (1, n_samples)
    audio_input = {"waveform": waveform, "sample_rate": 16000}

    diarization = pyannote_pipeline(audio_input, return_embeddings=True)
    print(f"[INFO] 화자분리 완료: \n {diarization}")

    diarization_names = diarization.speaker_diarization.labels()
    print(f"[DEBUG] pyannote 감지 화자 수: {len(diarization_names)}, 레이블: {diarization_names}")
    spk_id_map = {diarization_names[i]: f"spk_{i+1:02d}" for i in range(len(diarization_names))}
    spk_name_map = {f"spk_{i+1:02d}": f"알 수 없음 Speaker {i+1}" for i in range(len(diarization_names))}

    i = 0
    for e in diarization.speaker_embeddings:
        # 노름이 0인 제로 임베딩(실제 발화 없는 허상 화자) → cosine 계산 불가, 스킵
        if np.linalg.norm(e) < 1e-6:
            print(f"[DEBUG] 화자 {diarization_names[i]}: 제로 임베딩, 스킵")
            i += 1
            continue

        spk_id = f"spk_{i+1:02d}"
        scores = []
        ids = []
        for id, emb in embedding_speakers.items():
            score = 1 - cosine(np.array(emb).flatten(), np.array(e).flatten())
            scores.append(score)
            ids.append(id)
            print(f"[DEBUG] 화자 {diarization_names[i]} vs {id}: 유사도={score:.4f}")
        if scores and max(scores) >= 0.5:
            idx = scores.index(max(scores))
            spk_name_map[spk_id] = ids[idx]
        i += 1

    diarization.speaker_diarization.rename_labels(spk_id_map, copy=False)

    # community-1은 DiarizeOutput.speaker_diarization으로 순회
    try:
        iterable = diarization.speaker_diarization
    except AttributeError:
        iterable = ((turn, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True))

    segments = []
    for turn, speaker in iterable:
        segments.append({
            "speaker": speaker, # spk_01, spk_02거나 embedding과 매칭된 이름
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    segments.sort(key=lambda s: s["start"])
    print(f"[DEBUG] 화자분리 세그먼트 수: {len(segments)}, 화자별: { {s['speaker'] for s in segments} }")
    return segments, spk_name_map

def merge_to_sentences(word_timestamps):
    if not word_timestamps:
        return []
    sentences = []
    buf_words = []
    buf_start = None
    prev_end = None
    for w in word_timestamps:
        # ── 묵음 간격 감지: 새 단어가 시작되기 전 pause ──
        if buf_words and prev_end is not None:
            gap = w["start"] - prev_end
            if gap >= _PAUSE_THRESHOLD:
                sentences.append({
                    "text": " ".join(buf_words).strip(),
                    "start": buf_start,
                    "end": prev_end,
                })
                buf_words = []
                buf_start = None
        if buf_start is None:
            buf_start = w["start"]
        buf_words.append(w["text"])
        prev_end = w["end"]
        # ── 구두점 감지 ──
        text_stripped = w["text"].rstrip()
        if text_stripped and text_stripped[-1] in _SENT_ENDERS:
            sentences.append({
                "text": " ".join(buf_words).strip(),
                "start": buf_start,
                "end": w["end"],
            })
            buf_words = []
            buf_start = None
            prev_end = None
    # 남은 미완성 문장
    if buf_words and buf_start is not None:
        sentences.append({
            "text": " ".join(buf_words).strip(),
            "start": buf_start,
            "end": word_timestamps[-1]["end"],
        })
    return sentences

def assign_speakers(
    word_timestamps: list[dict],
    diarization_segments: list[dict],
    margin: float = 0.1,
    min_segment_duration: float = 0.3,) -> list[dict]:

    valid_segments = [
        seg for seg in diarization_segments
        if (seg["end"] - seg["start"]) >= min_segment_duration
    ]

    result = []
    for w in word_timestamps:
        mid = (w["start"] + w["end"]) / 2
        for seg in valid_segments:
            if (seg["start"] - margin) <= mid <= (seg["end"] + margin):
                result.append({**w, "speaker": seg["speaker"]})
                break
        # 매칭 없으면 제외
    return result

def merge_speaker_utterances(speaker_words: list[dict]) -> list[dict]:
    if not speaker_words:
        return []

    utterances = []
    cur_speaker = speaker_words[0]["speaker"]
    cur_words = [speaker_words[0]["text"]]
    cur_start = speaker_words[0]["start"]
    cur_end = speaker_words[0]["end"]

    for w in speaker_words[1:]:
        if w["speaker"] == cur_speaker:
            cur_words.append(w["text"])
            cur_end = w["end"]
        else:
            utterances.append({
                "speaker": cur_speaker,
                "text": " ".join(cur_words).strip(),
                "start": cur_start,
                "end": cur_end,
            })
            cur_speaker = w["speaker"]
            cur_words = [w["text"]]
            cur_start = w["start"]
            cur_end = w["end"]

    utterances.append({
        "speaker": cur_speaker,
        "text": " ".join(cur_words).strip(),
        "start": cur_start,
        "end": cur_end,
    })
    return utterances

def offline_diarization(wav16k: np.ndarray, participants_embeddings) -> list[dict]:
    pipeline = pyannote_pipeline

    # pyannote는 (waveform, sample_rate) dict 형태로 메모리 입력 지원
    waveform = torch.from_numpy(wav16k).unsqueeze(0)  # (1, n_samples)
    audio_input = {"waveform": waveform, "sample_rate": 16000}

    diarization = pipeline(audio_input, return_embeddings=True)
    print(f"[INFO] 화자분리 완료: \n {diarization}")
    print(f"[INFO] 화자분리 세그먼트: {type(diarization.speaker_diarization)}")
    # 화자 라벨 → 번호 매핑
    segments = []

    # 등록된 임베딩과 비교
    embedding_speakers = participants_embeddings # {user_id: embedding}
    diarization_names = diarization.speaker_diarization.labels()
    
    spk_id_map = {diarization_names[i]: f"spk_{i+1:02d}" for i in range(len(diarization_names))}
    spk_name_map = {f"spk_{i+1:02d}": f"알 수 없음 Speaker {i+1}" for i in range(len(diarization_names))}

    i = 0
    for e in diarization.speaker_embeddings:
        # 노름이 0인 제로 임베딩(실제 발화 없는 허상 화자) → cosine 계산 불가, 스킵
        if np.linalg.norm(e) < 1e-6:
            print(f"[DEBUG] 화자 {diarization_names[i]}: 제로 임베딩, 스킵")
            i += 1
            continue

        spk_id = f"spk_{i+1:02d}"
        scores = []
        names = []
        for id, emb in embedding_speakers.items():
            score = 1 - cosine(np.array(emb).flatten(), np.array(e).flatten())
            scores.append(score)
            names.append(id)
            print(f"[DEBUG] 화자 {diarization_names[i]} vs {id}: 유사도={score:.4f}")
        if scores and max(scores) >= 0.5:
            idx = scores.index(max(scores))
            spk_name_map[spk_id] = names[idx]
        i += 1

    diarization.speaker_diarization.rename_labels(spk_id_map, copy=False)


    # community-1은 DiarizeOutput.speaker_diarization으로 순회
    try:
        iterable = diarization.speaker_diarization
    except AttributeError:
        # fallback: 기존 pyannote Annotation 객체인 경우
        iterable = ((turn, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True))

    for turn, speaker in iterable:
        segments.append({
            "speaker": speaker, # spk_01, spk_02거나 embedding과 매칭된 이름
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    segments.sort(key=lambda s: s["start"])
    return segments, spk_name_map

def offline_asr(wav16k: np.ndarray) -> str:
    """오프라인 전사: 전체 오디오를 배치 전사하여 텍스트 반환."""
    model = asr
    results = model.transcribe(
        audio=[(wav16k, 16000)],
        language="Korean"
    )
    return (results[0].text or "").strip()


def offline_asr_chunked(wav16k: np.ndarray, segments: list[dict]) -> list[dict]:
    """
    화자 세그먼트별로 ASR을 수행.
    인접한 같은 화자의 세그먼트를 병합하고, 각 구간의 오디오를 배치 전사.
    """
    if not segments:
        return []

    # 인접 동일 화자 병합
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and seg["start"] - prev["end"] < 0.5:
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # 너무 짧은 세그먼트 필터 (1.0초 미만)
    merged = [s for s in merged if s["end"] - s["start"] >= 1.0]
    if not merged:
        return []

    # 각 세그먼트의 오디오 청크 추출
    audio_chunks = []
    valid_segments = []
    for seg in merged:
        start_sample = int(seg["start"] * 16000)
        end_sample = int(seg["end"] * 16000)
        chunk = wav16k[start_sample:end_sample]
        if chunk.shape[0] < 16000:  # 1.0초 미만 스킵
            continue
        audio_chunks.append((chunk, 16000))
        valid_segments.append(seg)

    if not audio_chunks:
        return []

    # 배치 전사
    model = asr
    asr_results = model.transcribe(
        audio=audio_chunks,
        language="Korean",
    )

    results = []
    for seg, asr_res in zip(valid_segments, asr_results):
        text = (asr_res.text or "").strip()
        if text:
            results.append({
                "speaker_id": None,  # spk_name_map으로 caller가 해소
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
            })

    return results

def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def build_minutes(segments_with_text: list[dict], meeting_start_time) -> list[dict]:
    """회의록 형식으로 변환."""
    minutes = []
    i = 1
    for seg in segments_with_text:
        minutes.append({
            "seq": i,
            "speaker_id": seg["speaker_id"],
            "speaker_label": seg["speaker"],
            "timestamp": (meeting_start_time + timedelta(seconds=seg['start'])).strftime("%Y-%m-%dT%H:%M:%S"),
            "content": seg["text"],
            "confidence": None,  # ASR 모델에서 confidence 점수 제공 시 여기에 추가 가능
        })
        i += 1
    return minutes