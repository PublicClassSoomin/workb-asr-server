# 화자 분리 함수
import numpy as np
from core.config import config
from core.models import aligner, pyannote_pipeline, asr
from scipy.spatial.distance import cosine
import torch
from datetime import timedelta
from dataclasses import dataclass
from typing import Optional

MAX_ALIGN_SEC = config.WINDOW_SEC
_PAUSE_THRESHOLD = config._PAUSE_THRESHOLD
_SENT_ENDERS = frozenset("。．.！!？?")   # 구두점 기반 (보조)

# --------------------------------------------------------------------------- #
# Speaker identity data structures                                              #
# --------------------------------------------------------------------------- #

@dataclass
class SpeakerIdentity:
    """단일 화자의 식별 정보."""
    user_id: Optional[int]     # 등록된 참가자 ID, 없으면 None
    display_name: str          # 표시 이름 ("김철수" 또는 "알 수 없음 Speaker N")
    match_score: float         # 참가자 임베딩 매칭 점수 (0.0 ~ 1.0)
    matched: bool              # 등록 참가자와 매칭 여부


class SpeakerRegistry:
    """
    회의 전체에서 안정적인 speaker ID를 유지하는 레지스트리.

    슬라이딩 윈도우마다 pyannote가 새 레이블을 부여해도
    cosine similarity로 동일 화자를 추적한다.
    """

    _REGISTRY_THRESHOLD: float = 0.75    # 동일 화자 판정 임계값
    _PARTICIPANT_THRESHOLD: float = 0.50  # 등록 참가자 매칭 임계값

    def __init__(self) -> None:
        self._embeddings: dict = {}   # spk_id → 정규화 임베딩 (np.ndarray)
        self._identities: dict = {}   # spk_id → SpeakerIdentity
        self._next_idx: int = 1

    # ── internal helpers ──────────────────────────────────────────────────── #

    def _alloc_spk_id(self) -> str:
        spk_id = f"spk_{self._next_idx:02d}"
        self._next_idx += 1
        return spk_id

    @staticmethod
    def _normalize(arr: np.ndarray) -> "Optional[np.ndarray]":
        n = float(np.linalg.norm(arr))
        return arr / n if n >= 1e-6 else None

    def _best_registry_match(self, emb_norm: np.ndarray) -> tuple:
        """기존 레지스트리에서 가장 유사한 화자를 반환. (spk_id, score)"""
        best_id, best_score = None, self._REGISTRY_THRESHOLD - 1e-9
        for spk_id, reg_emb in self._embeddings.items():
            score = float(1.0 - cosine(emb_norm, reg_emb))
            if score > best_score:
                best_score, best_id = score, spk_id
        return best_id, best_score

    def _best_participant_match(
        self,
        emb_norm: np.ndarray,
        participants_embeddings: dict,
        participants_names: dict,
    ) -> tuple:
        """등록된 참가자 중 가장 유사한 화자를 반환. (user_id, score, display_name)"""
        best_uid, best_score = None, self._PARTICIPANT_THRESHOLD - 1e-9
        for uid, p_emb in participants_embeddings.items():
            p_arr = np.array(p_emb).flatten()
            if p_arr.size == 0:
                continue
            p_norm = self._normalize(p_arr)
            if p_norm is None:
                continue
            score = float(1.0 - cosine(emb_norm, p_norm))
            if score > best_score:
                best_score, best_uid = score, uid
        if best_uid is not None:
            return best_uid, best_score, participants_names.get(best_uid, f"Speaker {best_uid}")
        return None, 0.0, ""

    # ── public API ─────────────────────────────────────────────────────────── #

    def resolve_window(
        self,
        pyannote_labels: list,
        pyannote_embeddings,
        participants_embeddings: dict,
        participants_names: dict,
    ) -> dict:
        """
        pyannote 레이블 + 임베딩을 안정적인 spk_id로 매핑.
        레지스트리를 갱신하고 {pyannote_label: spk_id} 를 반환.
        """
        label_to_spk: dict = {}

        for label, raw_emb in zip(pyannote_labels, pyannote_embeddings):
            emb_arr = np.array(raw_emb).flatten()
            emb_norm = self._normalize(emb_arr)
            if emb_norm is None:
                print(f"[DEBUG] pyannote 레이블 {label}: 제로 임베딩, 스킵")
                continue

            existing_id, reg_score = self._best_registry_match(emb_norm)

            if existing_id is not None:
                # 동일 화자 발견 → rolling-average 임베딩 업데이트
                alpha = 0.9
                updated = alpha * self._embeddings[existing_id] + (1.0 - alpha) * emb_norm
                normed = self._normalize(updated)
                if normed is not None:
                    self._embeddings[existing_id] = normed

                # 아직 참가자 미매칭이면 재시도
                if not self._identities[existing_id].matched:
                    uid, score, name = self._best_participant_match(
                        emb_norm, participants_embeddings, participants_names
                    )
                    if uid is not None:
                        self._identities[existing_id] = SpeakerIdentity(
                            user_id=uid,
                            display_name=name,
                            match_score=round(score, 4),
                            matched=True,
                        )

                label_to_spk[label] = existing_id
                print(f"[DEBUG] {label} → 기존 {existing_id} (유사도={reg_score:.4f})")
            else:
                # 신규 화자 등록
                new_id = self._alloc_spk_id()
                self._embeddings[new_id] = emb_norm

                uid, score, name = self._best_participant_match(
                    emb_norm, participants_embeddings, participants_names
                )
                if uid is not None:
                    self._identities[new_id] = SpeakerIdentity(
                        user_id=uid,
                        display_name=name,
                        match_score=round(score, 4),
                        matched=True,
                    )
                else:
                    spk_num = self._next_idx - 1
                    self._identities[new_id] = SpeakerIdentity(
                        user_id=None,
                        display_name=f"알 수 없음 Speaker {spk_num}",
                        match_score=0.0,
                        matched=False,
                    )

                label_to_spk[label] = new_id
                print(f"[DEBUG] {label} → 신규 {new_id} 등록")

        return label_to_spk

    @property
    def identities(self) -> dict:
        """현재 레지스트리의 {spk_id: SpeakerIdentity} 복사본 반환."""
        return dict(self._identities)

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
    
def run_diarization(
    wav16k: np.ndarray,
    participants_embeddings: dict,
    participants_names: dict,
    registry: SpeakerRegistry,
) -> tuple[list[dict], dict]:
    """
    화자분리를 수행하여 화자 세그먼트 리스트와 speaker identity map을 반환.

    SpeakerRegistry를 통해 슬라이딩 윈도우 간 화자 ID가 안정적으로 유지된다.

    Returns: (segments, identity_map)
      - segments: [{"speaker": "spk_01", "start": 0.0, "end": 2.5}, ...]
      - identity_map: {"spk_01": SpeakerIdentity(...), ...}
    """
    waveform = torch.from_numpy(wav16k).unsqueeze(0)  # (1, n_samples)
    audio_input = {"waveform": waveform, "sample_rate": 16000}

    diarization = pyannote_pipeline(audio_input, return_embeddings=True)

    pyannote_labels = diarization.speaker_diarization.labels()
    print(f"[DEBUG] pyannote 감지 화자 수: {len(pyannote_labels)}, 레이블: {pyannote_labels}")

    # registry를 통해 안정적인 spk_id 매핑 획득
    label_to_spk = registry.resolve_window(
        pyannote_labels,
        list(diarization.speaker_embeddings),
        participants_embeddings,
        participants_names,
    )

    # pyannote 레이블을 안정적인 spk_id로 rename
    rename_map = {lbl: label_to_spk[lbl] for lbl in pyannote_labels if lbl in label_to_spk}
    if rename_map:
        diarization.speaker_diarization.rename_labels(rename_map, copy=False)

    # community-1은 DiarizeOutput.speaker_diarization으로 순회
    try:
        iterable = diarization.speaker_diarization
    except AttributeError:
        iterable = ((turn, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True))

    valid_spk_ids = set(label_to_spk.values())
    segments = []
    for turn, speaker in iterable:
        # 제로 임베딩으로 스킵된 레이블은 segments에서 제외
        if speaker not in valid_spk_ids:
            continue
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    segments.sort(key=lambda s: s["start"])
    identity_map = registry.identities
    print(f"[DEBUG] 화자분리 세그먼트 수: {len(segments)}, 화자별: { {s['speaker'] for s in segments} }")
    return segments, identity_map

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
    margin: float = 0.0,
    min_segment_duration: float = 1.0,
) -> list[dict]:
    """
    각 단어에 화자를 할당한다.

    midpoint 기준 대신 word-segment overlap duration을 계산하여
    가장 많이 겹치는 화자를 선택한다.
    매칭 실패 시 직전 화자(fallback)를 사용하고,
    직전 화자도 없으면 "unknown"으로 처리한다.
    """
    valid_segments = [
        seg for seg in diarization_segments
        if (seg["end"] - seg["start"]) >= min_segment_duration
    ]

    result: list[dict] = []
    last_speaker: Optional[str] = None

    for w in word_timestamps:
        word_dur = max(w["end"] - w["start"], 1e-6)
        best_spk: Optional[str] = None
        best_overlap = 0.0

        for seg in valid_segments:
            seg_start = seg["start"] - margin
            seg_end   = seg["end"]   + margin
            overlap = max(0.0, min(w["end"], seg_end) - max(w["start"], seg_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = seg["speaker"]

        overlap_ratio = round(best_overlap / word_dur, 3)

        if best_spk is not None:
            last_speaker = best_spk
            result.append({**w, "speaker": best_spk, "overlap_ratio": overlap_ratio})
        elif last_speaker is not None:
            # fallback: 직전 화자
            result.append({**w, "speaker": last_speaker, "overlap_ratio": 0.0})
        else:
            result.append({**w, "speaker": "unknown", "overlap_ratio": 0.0})

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

def offline_diarization(
    wav16k: np.ndarray,
    participants_embeddings: dict,
    participants_names: dict,
    registry: SpeakerRegistry,
) -> tuple[list[dict], dict]:
    """
    오프라인 전체 오디오 화자분리 (streaming 종료 후 실행).

    streaming 단계에서 쌓인 registry를 재사용하므로
    spk_id가 실시간 단계와 일관되게 유지된다.
    """
    print("[INFO] 오프라인 화자분리 시작...")
    segments, identity_map = run_diarization(
        wav16k, participants_embeddings, participants_names, registry
    )
    print(f"[INFO] 오프라인 화자분리 완료: {len(segments)}개 세그먼트")
    return segments, identity_map

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
    
    FILTER_SEC = 0.8
    MERGE_SEC = 0.5

    # 인접 동일 화자 병합
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and seg["start"] - prev["end"] < MERGE_SEC:
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # 너무 짧은 세그먼트 필터 (1초 미만)
    merged = [s for s in merged if s["end"] - s["start"] >= FILTER_SEC]
    if not merged:
        return []

    # 각 세그먼트의 오디오 청크 추출
    audio_chunks = []
    valid_segments = []
    for seg in merged:
        start_sample = int(seg["start"] * 16000)
        end_sample = int(seg["end"] * 16000)
        chunk = wav16k[start_sample:end_sample]
        if chunk.shape[0] < 16000 * FILTER_SEC:  # 1초 미만 스킵
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
            "start": seg['start'],
            "end": seg['end'],
            "confidence": None,  # ASR 모델에서 confidence 점수 제공 시 여기에 추가 가능
        })
        i += 1
    return minutes