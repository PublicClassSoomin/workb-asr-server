# 라우팅
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from core.config import config
from core.models import asr, aligner, pyannote_pipeline
from db.mysql import get_participants, get_participants_embeddings, get_meeting_info, save_user_embedding
import threading
from services.audio_utils import bytes_to_wav16k
from services.diarization import align_chunk, assign_speakers, merge_speaker_utterances,run_diarization, merge_to_sentences, offline_diarization, offline_asr_chunked, offline_asr, build_minutes
from services.text_util import fix_spacing_with_kiwi
import numpy as np
import torch
import json
import soundfile as sf
import os
from datetime import datetime, timezone, timedelta
from db.redis_client import get_redis
from db.mongodb import upload_mongodb


import re
from pathlib import Path
from kiwipiepy import Kiwi

STORAGE_ROOT = Path(__file__).parent.parent / "src"

_kiwi = Kiwi()

OVERLAP_SEC = config.OVERLAP_SEC
WINDOW_SEC = config.WINDOW_SEC
ALIGNER_MODEL_PATH = config.ALIGNER_MODEL_PATH
DIARIZE_MODEL_PATH = config.DIARIZE_MODEL_PATH
HF_TOKEN = config.HF_TOKEN
REDIS_TTL_SEC=config.REDIS_TTL_SEC
DIARIZE_WINDOW_SEC = 600  # 화자분리 슬라이딩 윈도우 상한 (10분)

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

@router.get("/{meeting_id}/audio")
async def get_meeting_audio(meeting_id: str):
    """
    회의 오디오 파일 스트리밍.
    브라우저 Audio가 Range 요청으로 원하는 구간(start~end)만 읽음.
    """
    audio_path = STORAGE_ROOT / f"meeting_{meeting_id}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="오디오 파일이 없습니다.")

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename="audio.wav",
        headers={"Accept-Ranges": "bytes"},  # 구간 재생에 필수
    )

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
    participants_embeddings = await get_participants_embeddings(p_ids)  # {user_id: embedding} 딕셔너리
    print(f"Participants: {participants}, Embeddings fetched for IDs: {list(participants_embeddings.keys())}")

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
    diarize_history_spk_map: dict = {}    # 10분 window 초과 시 누적된 화자 맵
    spk_id_map: dict = {}                 # spk_xx → userId (Redis 저장용)

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

                    # ── 슬라이딩마다 화자분리 수행 (최근 10분 window) ──
                    try:
                        total_dur = total_received_samples / 16000

                        if total_dur > DIARIZE_WINDOW_SEC:
                            # 최근 10분에 해당하는 청크만 뒤에서부터 모아 concat
                            win_samples = int(DIARIZE_WINDOW_SEC * 16000)
                            selected_chunks = []
                            collected = 0
                            for chunk in reversed(full_audio_chunks):
                                selected_chunks.append(chunk)
                                collected += chunk.shape[0]
                                if collected >= win_samples:
                                    break
                            selected_chunks.reverse()
                            window_audio = np.concatenate(selected_chunks)[-win_samples:]
                            time_offset = (total_received_samples - win_samples) / 16000

                            new_segs, new_spk_map = run_diarization(window_audio, participants_embeddings)

                            # 절대 시간으로 보정
                            for seg in new_segs:
                                seg["start"] = round(seg["start"] + time_offset, 3)
                                seg["end"] = round(seg["end"] + time_offset, 3)

                            # window 이전 세그먼트는 이전 결과 재사용
                            old_segs = [s for s in diarization_segments if s["end"] <= time_offset]
                            diarization_segments = old_segs + new_segs

                            # 화자 맵: 누적 히스토리 + 새 결과 병합 (새 결과 우선)
                            diarize_history_spk_map = {**diarize_history_spk_map, **new_spk_map}
                            spk_name_map = diarize_history_spk_map
                        else:
                            slide_audio = np.concatenate(full_audio_chunks)
                            diarization_segments, spk_name_map = run_diarization(slide_audio, participants_embeddings)
                            diarize_history_spk_map = dict(spk_name_map)

                        # spk_xx → userId 매핑 저장 (이름 변환 전, Redis 저장용)
                        spk_id_map = {spk: uid for spk, uid in spk_name_map.items() if isinstance(uid, int)}

                        # spk_name_map을 실제 이름으로 업데이트
                        for spk, name in spk_name_map.items():
                            if type(name) == int:
                                spk_name_map[spk] = participants[name]
                        # 이름 변환 결과를 히스토리에도 반영
                        diarize_history_spk_map.update(spk_name_map)
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
                    if spk_id_map:
                        await _r.hset(f"meeting:{meeting_id}:speakers", mapping={k: str(v) for k, v in spk_id_map.items()})
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

        # 전체 오디오 src 폴더에 저장
        src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
        os.makedirs(src_dir, exist_ok=True)
        audio_path = os.path.join(src_dir, f"meeting_{meeting_id}.wav")
        sf.write(audio_path, wav16k, 16000)
        print(f"[INFO] 오디오 저장 완료: {audio_path}")

        # 1) 화자분리 (CPU — GPU 점유 없음)
        diarize_segments, spk_name_map = offline_diarization(wav16k, participants_embeddings)

        # 2) 화자 세그먼트별 ASR
        segments_with_text = offline_asr_chunked(wav16k, diarize_segments)

        # 3) spk_name_map으로 speaker_id 해소 (spk_id → user_id)
        # seg["speaker"]는 항상 "spk_01" 형태의 문자열이므로 spk_name_map에서 resolve
        for seg in segments_with_text:
            spk_id = seg["speaker"]          # "spk_01" 등 문자열
            resolved = spk_name_map.get(spk_id)  # int(user_id) or "알 수 없음 ..." or None
            if isinstance(resolved, int):
                seg["speaker_id"] = resolved
                seg["speaker"] = participants.get(resolved, spk_id)
            else:
                seg["speaker_id"] = None
                seg["speaker"] = resolved if resolved else spk_id

        # 4) 세그먼트가 없으면 넘어가
        if not segments_with_text:
            print("No segments with text were generated, skipping MongoDB upload.")
            await ws.close()
            return

    minutes = build_minutes(segments_with_text, meeting_start_time)

    # 몽고db에 저장
    upload_mongodb(minutes, meeting_id, workspace_id, duration, meeting_start_time)
    print(f"Meeting {meeting_id} processed and uploaded to MongoDB with duration {duration} seconds.")

    await ws.send_json({"message": "Meeting processing complete", "meeting_id": meeting_id})
    await ws.close()

@router.post("/embedding/{user_id}")
async def update_embedding(user_id: int, audio: UploadFile = File(...)):
    # FormData의 audio 필드로 WAV 파일을 받아 임베딩 등록
    # 앞뒤로 2초씩 자르기
    audio_bytes = await audio.read()
    wav16k = bytes_to_wav16k(audio_bytes)
    wav16k = wav16k[32000:-32000] if wav16k.shape[0] > 64000 else wav16k

    embedding = pyannote_pipeline._embedding
    # (batch=1, channel=1, samples) 형태의 torch tensor로 변환
    waveform = torch.from_numpy(wav16k).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    emb = embedding(waveform)
    emb_np = emb.squeeze(0) if hasattr(emb, "squeeze") else emb
    print(emb_np, emb_np.shape)
    emb_str = str(emb_np.tolist())
    print(emb_str)
    # DB에 저장
    # 기존에 있던 임베딩은 덮어쓰기
    await save_user_embedding(user_id, emb_str)
    return {"message": f"성공적으로 저장되었습니다."}






