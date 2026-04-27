# 라우팅
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.config import config
from core.models import asr, aligner, pyannote_pipeline
from db.mysql import get_participants, get_participants_embeddings, get_meeting_info
import threading
from services.audio_utils import bytes_to_wav16k
from services.diarization import align_chunk, assign_speakers, merge_speaker_utterances,run_diarization, merge_to_sentences, offline_diarization, offline_asr_chunked, offline_asr, build_minutes
from services.text_util import fix_spacing_with_kiwi
import numpy as np
import json
from datetime import datetime, timezone, timedelta
from db.redis_client import get_redis
from db.mongodb import upload_mongodb


import re
from kiwipiepy import Kiwi

_kiwi = Kiwi()

OVERLAP_SEC = config.OVERLAP_SEC
WINDOW_SEC = config.WINDOW_SEC
ALIGNER_MODEL_PATH = config.ALIGNER_MODEL_PATH
DIARIZE_MODEL_PATH = config.DIARIZE_MODEL_PATH
HF_TOKEN = config.HF_TOKEN
REDIS_TTL_SEC=config.REDIS_TTL_SEC

asr_lock = threading.Lock()

_ASR_HALLUCINATION_PATTERN = re.compile(
    r"(system|user|assistant|language\s*\w+|asr\s*text)",
    re.IGNORECASE,
)

def _clean_content(text: str) -> str:
    """ASR 할루시네이션 아티팩트 제거 및 띄어쓰기 교정"""
    text = _ASR_HALLUCINATION_PATTERN.sub("", text).strip()
    if not text:
        return ""
    return fix_spacing_with_kiwi(text)


router = APIRouter(prefix="/meeting")

@router.get("/{meeting_id}")
def get_meeeting(meeting_id: str):
    return {"meeting_id": meeting_id}

@router.websocket("/ws/stream/{meeting_id}")
async def ws_meeting(ws: WebSocket, meeting_id: str):
    await ws.accept()
    meeting_start_time = datetime.now(timezone.utc) + timedelta(hours=9)

    # receive_json()이 실패할 경우(클라이언트가 바이트를 먼저 보내는 등) 기본값 사용
    language = "Korean"
    try:
        first = await ws.receive_json()
        language = first.get("language", "Korean")
    except Exception:
        print("Failed to receive JSON, using default language:", language)

    meeting_id = meeting_id
    participants = await get_participants(meeting_id)  # id:name 딕셔너리
    meeting_info = await get_meeting_info(meeting_id)
    workspace_id = meeting_info["workspace_id"] if meeting_info else None
    p_ids = list(participants.keys())
    p_names = list(participants.values())
    participants_embeddings = await get_participants_embeddings(p_ids)
    participants_embeddings = dict(zip(p_ids, participants_embeddings))  # id:embedding 딕셔너리

    # ── ASR 스트리밍 상태 ─────────────────────────────────
    accumulated_text = ""   # 슬라이드로 확정된 이전 세션들의 텍스트
    overlap_text_len = 0    # 현재 세션에서 오버랩 재투입으로 생성된 텍스트 길이
    max_samples = int(30 * 16000)
    overlap_samples = int(OVERLAP_SEC * 16000)


    full_audio_chunks = []              # 수신 오디오 순차 축적
    committed_timestamps = []           # 확정 정렬 타임스탬프 (불변)
    committed_chunk_count = 0           # full_audio_chunks 중 확정 완료 chunk 수
    last_align_chunk_idx = 0            # 마지막 정렬 완료 chunk 인덱스
    last_align_text_len = 0             # 마지막 정렬 완료 텍스트 길이
    last_align_time_offset = 0.0        # 다음 정렬 청크의 절대 시간 오프셋 (초)
    total_received_samples = 0          # 수신 오디오 총 샘플 수
    ts_dirty = False                    # 새 타임스탬프 발생 시 True

    # ── 화자분리 상태 ──────────────────────────────────────
    diarization_segments: list[dict] = []  # 최신 pyannote 화자분리 결과
    spk_name_map: dict = {}               # spk_XX → 실제 이름
    last_diarize_samples = 0               # 마지막 화자분리 실행 시점의 누적 샘플 수
    diarize_dirty = False              # 새 화자분리 결과 발생 시 True

    with asr_lock:
        state = asr.init_streaming_state(
            language=language,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        )
        
        try:
            while True:
                data = await ws.receive_bytes()

                # ── 종료 신호: 빈 바이트 수신 ─────────────────
                if len(data) == 0:
                    break

                # ── 오디오 전처리 및 스트리밍 추론 ───────────
                wav16k = bytes_to_wav16k(data)
                full_audio_chunks.append(wav16k)
                total_received_samples += wav16k.shape[0]
                asr.streaming_transcribe(wav16k, state)

                # ── 슬라이딩 윈도우 처리 ──────────────────────
                if state.audio_accum.shape[0] >= max_samples:
                    max_samples = int(WINDOW_SEC * 16000)  # 윈도우 크기만큼 슬라이드
                    asr.finish_streaming_transcribe(state)

                    # ① 확정: 오버랩 제외 텍스트를 accumulated_text에 커밋
                    full_text = state.text or ""
                    new_committed = full_text[overlap_text_len:]
                    if new_committed:
                        accumulated_text += new_committed + " "

                    # ② 오버랩용 오디오 추출 (마지막 OVERLAP_SEC)
                    overlap_audio = state.audio_accum[-overlap_samples:]

                    # ③ 새 세션 초기화 + 오버랩 재투입
                    state = asr.init_streaming_state(
                        language=language,
                        unfixed_chunk_num=2,
                        unfixed_token_num=5,
                        chunk_size_sec=2.0,
                    )
                    asr.streaming_transcribe(overlap_audio, state)
                    overlap_text_len = len(state.text or "")

                    # ── 확정 타임스탬프 정렬 (committed) ─────
                    #   슬라이드로 텍스트가 확정되었으므로
                    #   새로 확정된 오디오+텍스트를 정렬한다.
                    #   확정 텍스트는 불변이므로 결과도 불변.
                    committed_chunk_count = len(full_audio_chunks)
                    new_chunks = full_audio_chunks[last_align_chunk_idx:committed_chunk_count]
                    new_text = accumulated_text[last_align_text_len:].strip()
                    if new_chunks and new_text:
                        new_ts = align_chunk(
                            new_chunks, new_text,
                            state.language or language,
                            last_align_time_offset,
                        )
                        committed_timestamps.extend(new_ts)
                        chunk_samples = sum(c.shape[0] for c in new_chunks)
                        last_align_time_offset += chunk_samples / 16000
                        last_align_chunk_idx = committed_chunk_count
                        last_align_text_len = len(accumulated_text)
                    ts_dirty = True

                    # ── 슬라이딩마다 화자분리 수행 ───────────────
                    try:
                        slide_audio = np.concatenate(full_audio_chunks)
                        diarization_segments, spk_name_map = run_diarization(
                            slide_audio,
                            participants_embeddings,
                        )
                        # spk_name_map을 실제 이름으로 업데이트
                        i = 0
                        for spk, name in spk_name_map.items():
                            if type(name) == int:
                                spk_name_map[spk] = participants[name]
                        diarize_dirty = True
                    except Exception as _exc:
                        print(f"[ERROR] sliding diarization: {_exc}")

                # ── 응답 전송 ─────────────────────────────────
                current_session_text = state.text or ""
                display_new = (
                    current_session_text[overlap_text_len:]
                    if len(current_session_text) > overlap_text_len
                    else current_session_text
                )
                # display_text = (accumulated_text + display_new).strip()

                response = {
                    "language": state.language or language,
                    "text": display_new.strip(),
                    "final": False,
                }
                if ts_dirty:
                    # all_ts = committed_timestamps
                    # response["sentences"] = merge_to_sentences(all_ts)
                    ts_dirty = False
                if diarize_dirty and committed_timestamps and diarization_segments:
                    _r = get_redis()
                    await _r.hset(f"meeting:{meeting_id}:speakers", mapping=spk_name_map)
                    await _r.expire(f"meeting:{meeting_id}:speakers", REDIS_TTL_SEC)
                    speaker_words = assign_speakers(committed_timestamps, diarization_segments)
                    utterances = merge_speaker_utterances(speaker_words)
                    await _r.delete(f"meeting:{meeting_id}:utterances")
                    for utt in utterances:
                        ts = (meeting_start_time + timedelta(seconds=utt["start"])).strftime("%Y-%m-%dT%H:%M:%S")
                        content = _clean_content(utt["text"])
                        if not content:
                            continue
                        await _r.rpush(f"meeting:{meeting_id}:utterances", json.dumps({
                            "speaker_id": utt["speaker"],
                            "content": content,
                            "timestamp": ts,
                        }, ensure_ascii=False))
                    await _r.expire(f"meeting:{meeting_id}:utterances", REDIS_TTL_SEC)
                    raw_utts = await _r.lrange(f"meeting:{meeting_id}:utterances", 0, -1)
                    diarization_resp = []
                    for raw in raw_utts:
                        item = json.loads(raw)
                        item["speaker"] = spk_name_map.get(item["speaker_id"], item["speaker_id"])
                        diarization_resp.append(item)
                    response["diarization"] = diarization_resp
                    diarize_dirty = False
                await ws.send_json(response)
                await get_redis().set(f"meeting:{meeting_id}:latest", display_new.strip(), ex=REDIS_TTL_SEC)

        except (WebSocketDisconnect, RuntimeError):
            if asr:
                asr.finish_streaming_transcribe(state)

        # ── 최종 처리 ─────────────────────────────────────────
        asr.finish_streaming_transcribe(state)
        
        wav16k = np.concatenate(full_audio_chunks)
        duration = round(wav16k.shape[0] / 16000, 2)

        # 1) 화자분리 (CPU — GPU 점유 없음)
        diarize_segments, spk_name_map = offline_diarization(wav16k, participants_embeddings)

        # 2) 화자 세그먼트별 ASR
        segments_with_text = offline_asr_chunked(wav16k, diarize_segments)

        # 3) spk_name_map으로 speaker_id 해소 (spk_id → user_id)
        for seg in segments_with_text:
            spk_id = seg["speaker"]
            if type(spk_id) == int:
                user_id = spk_name_map.get(spk_id)  # 매칭된 경우 user_id, 아니면 None
            else:
                user_id = None  # spk_id가 이미 이름인 경우
            seg["speaker_id"] = user_id
            seg["speaker"] = participants.get(user_id, spk_id) if user_id else spk_id

        # 4) 세그먼트가 없으면 넘어가
        if not segments_with_text:
            print("No segments with text were generated, skipping MongoDB upload.")
            ws.close()
            return

    minutes = build_minutes(segments_with_text, meeting_start_time)

    # 몽고db에 저장
    upload_mongodb(minutes, meeting_id, workspace_id, duration, meeting_start_time)
    print(f"Meeting {meeting_id} processed and uploaded to MongoDB with duration {duration} seconds.")

    await ws.send_json({"message": "Meeting processing complete", "meeting_id": meeting_id})


