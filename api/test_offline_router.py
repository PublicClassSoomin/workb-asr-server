from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from threading import Lock

import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from core.config import config
from db.mongodb import get_db
from services.audio_utils import bytes_to_wav16k, preprocess_audio_bytes

router = APIRouter(prefix="/meeting-test", tags=["meeting-test"])

_diarizer = None
_diarizer_lock = Lock()
_collection_name_pattern = re.compile(r"^[A-Za-z0-9_.-]+$")
_kst = timezone(timedelta(hours=9))


def _kst_now() -> datetime:
    return datetime.now(_kst)


def _normalize_meeting_start_time(value: datetime | None) -> datetime:
    if value is None:
        return _kst_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=_kst)
    return value.astimezone(_kst)


def _coalesce_param(query_value: str | None, form_value: str | None) -> str | None:
    if query_value is not None and query_value != "":
        return query_value
    if form_value is not None and form_value != "":
        return form_value
    return None


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="meeting_start_time 형식이 올바르지 않습니다. ISO datetime 문자열을 사용하세요.",
        ) from exc


def _parse_document_fields(document_fields: str | None) -> dict:
    if not document_fields:
        return {}

    try:
        parsed = json.loads(document_fields)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"document_fields는 JSON object 문자열이어야 합니다: {exc.msg}",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="document_fields는 JSON object여야 합니다.")

    reserved_keys = {
        "meeting_id",
        "workspace_id",
        "meeting_start_time",
        "created_at",
        "updated_at",
        "total_duration_sec",
        "audio_filename",
        "audio_content_type",
        "preprocessing_applied",
        "preprocessing_warning",
        "diarization_segment_count",
        "diarization_segments",
    }
    overlap = reserved_keys.intersection(parsed.keys())
    if overlap:
        overlap_text = ", ".join(sorted(overlap))
        raise HTTPException(
            status_code=400,
            detail=f"document_fields에 예약 필드를 덮어쓸 수 없습니다: {overlap_text}",
        )

    return parsed


def _get_diarizer():
    global _diarizer

    if _diarizer is not None:
        return _diarizer

    with _diarizer_lock:
        if _diarizer is None:
            if not config.DIARIZE_MODEL_PATH:
                raise RuntimeError("DIARIZE_MODEL_PATH 환경변수가 설정되지 않았습니다.")

            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(
                config.DIARIZE_MODEL_PATH,
                token=config.HF_TOKEN or None,
            )
            pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            _diarizer = pipeline

    return _diarizer


def _run_offline_diarization(wav16k: np.ndarray) -> list[dict]:
    diarizer = _get_diarizer()
    waveform = torch.from_numpy(np.asarray(wav16k, dtype=np.float32)).unsqueeze(0)
    diarization = diarizer({"waveform": waveform, "sample_rate": 16000})

    raw_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_segments.append(
            {
                "speaker_raw_label": speaker,
                "start_sec": round(float(turn.start), 3),
                "end_sec": round(float(turn.end), 3),
            }
        )

    raw_segments.sort(key=lambda item: (item["start_sec"], item["end_sec"]))

    speaker_aliases: dict[str, str] = {}
    diarization_segments: list[dict] = []
    for index, segment in enumerate(raw_segments, start=1):
        raw_label = segment["speaker_raw_label"]
        speaker_label = speaker_aliases.setdefault(
            raw_label,
            f"speaker_{len(speaker_aliases) + 1:02d}",
        )
        diarization_segments.append(
            {
                "seq": index,
                "speaker_label": speaker_label,
                "speaker_raw_label": raw_label,
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "duration_sec": round(segment["end_sec"] - segment["start_sec"], 3),
            }
        )

    return diarization_segments


@router.post("/offline-diarization")
async def offline_diarization_test(
    meeting_id: str | None = Query(None, description="MongoDB에 저장할 meeting_id"),
    workspace_id: str | None = Query(None, description="MongoDB에 저장할 workspace_id"),
    meeting_start_time: str | None = Query(
        None,
        description="회의 시작 시각 ISO 문자열. 없으면 현재 KST 시각 사용",
    ),
    collection_name: str | None = Query(
        None,
        description="저장할 MongoDB collection 이름",
    ),
    document_fields: str | None = Query(
        None,
        description="추가로 저장할 MongoDB 필드(JSON object 문자열)",
    ),
    meeting_id_form: str | None = Form(None, alias="meeting_id"),
    workspace_id_form: str | None = Form(None, alias="workspace_id"),
    meeting_start_time_form: str | None = Form(None, alias="meeting_start_time"),
    collection_name_form: str | None = Form(None, alias="collection_name"),
    document_fields_form: str | None = Form(None, alias="document_fields"),
    audio: UploadFile = File(...),
):
    meeting_id_value = _coalesce_param(meeting_id, meeting_id_form)
    if meeting_id_value is None:
        raise HTTPException(
            status_code=422,
            detail="meeting_id는 query string 또는 form-data로 전달해야 합니다.",
        )

    workspace_id_value = _coalesce_param(workspace_id, workspace_id_form)
    meeting_start_time_value = _parse_optional_datetime(
        _coalesce_param(meeting_start_time, meeting_start_time_form)
    )
    collection_name_value = _coalesce_param(collection_name, collection_name_form) or "utterances"
    document_fields_value = _coalesce_param(document_fields, document_fields_form)

    if not _collection_name_pattern.fullmatch(collection_name_value):
        raise HTTPException(status_code=400, detail="collection_name 형식이 올바르지 않습니다.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="업로드된 오디오 파일이 비어 있습니다.")

    try:
        raw_wav16k = bytes_to_wav16k(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"오디오 디코딩에 실패했습니다: {exc}") from exc

    if raw_wav16k.size == 0:
        raise HTTPException(status_code=400, detail="변환된 오디오 길이가 0입니다.")

    preprocessing_applied = False
    preprocessing_warning = None
    try:
        wav16k = preprocess_audio_bytes(audio_bytes)
        preprocessing_applied = True
    except Exception as exc:
        wav16k = raw_wav16k
        preprocessing_warning = str(exc)

    if wav16k.size == 0:
        raise HTTPException(status_code=400, detail="전처리 후 오디오 길이가 0입니다.")

    try:
        diarization_segments = _run_offline_diarization(wav16k)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"오프라인 화자분리에 실패했습니다: {exc}") from exc

    normalized_start_time = _normalize_meeting_start_time(meeting_start_time_value)
    diarization_segments_with_time = []
    for segment in diarization_segments:
        timestamp = normalized_start_time + timedelta(seconds=segment["start_sec"])
        diarization_segments_with_time.append(
            {
                **segment,
                "timestamp": timestamp.isoformat(timespec="seconds"),
            }
        )

    duration_sec = round(float(wav16k.shape[0]) / 16000.0, 3)
    now = _kst_now()
    extra_document_fields = _parse_document_fields(document_fields_value)
    document = {
        "meeting_id": meeting_id_value,
        "workspace_id": workspace_id_value,
        "meeting_start_time": normalized_start_time,
        "created_at": now,
        "updated_at": now,
        "total_duration_sec": duration_sec,
        "audio_filename": audio.filename,
        "audio_content_type": audio.content_type,
        "preprocessing_applied": preprocessing_applied,
        "preprocessing_warning": preprocessing_warning,
        "diarization_segment_count": len(diarization_segments_with_time),
        "diarization_segments": diarization_segments_with_time,
    }
    document.update(extra_document_fields)

    result = await get_db()[collection_name_value].insert_one(document)

    return {
        "message": "offline diarization test complete",
        "collection": collection_name_value,
        "inserted_id": str(result.inserted_id),
        "meeting_id": meeting_id_value,
        "total_duration_sec": duration_sec,
        "preprocessing_applied": preprocessing_applied,
        "preprocessing_warning": preprocessing_warning,
        "diarization_segment_count": len(diarization_segments_with_time),
        "diarization_segments": diarization_segments_with_time,
    }