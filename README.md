# ASR Server — 프로젝트 명세서

## 1. 개요

실시간 회의 음성을 수신하여 **STT(음성→텍스트) + 화자분리(Speaker Diarization)** 를 수행하고,
결과를 Redis에 즉시 스트리밍하면서 회의 종료 후 최종 회의록을 MongoDB에 저장하는 백엔드 서버.

---

## 2. 기술 스택

| 분류           | 라이브러리 / 서비스              |
| -------------- | -------------------------------- |
| 웹 프레임워크  | FastAPI + Uvicorn                |
| 실시간 통신    | WebSocket                        |
| ASR 모델       | Qwen3ASRModel (vLLM 기반)        |
| 강제 정렬      | Qwen3ForcedAligner               |
| 화자분리       | pyannote.audio Pipeline          |
| 텍스트 후처리  | kiwipiepy (한국어 띄어쓰기 교정) |
| 오디오 처리    | soundfile, numpy, ffmpeg         |
| 주 DB (구조화) | MySQL (aiomysql)                 |
| 캐시 / 실시간  | Redis (redis.asyncio)            |
| 최종 저장      | MongoDB (motor)                  |

---

## 3. 디렉터리 구조

```
asr_server/
├── main.py                  # FastAPI 애플리케이션 진입점
├── .env                     # 환경변수 설정 파일
├── api/
│   └── router.py            # WebSocket 및 REST 라우터
├── core/
│   ├── config.py            # 환경변수 로드 및 Config 객체
│   └── models.py            # ASR / Aligner / Diarizer 모델 로더
├── db/
│   ├── mysql.py             # MySQL 커넥션 풀 및 쿼리 함수
│   ├── mongodb.py           # MongoDB 연결 및 회의록 저장
│   └── redis_client.py      # Redis 연결 및 클라이언트 제공
└── services/
    ├── audio_utils.py       # 오디오 전처리 (리샘플링, WebM/Opus 변환)
    ├── diarization.py       # 화자분리, 타임스탬프 정렬, 발화 병합, 오프라인 ASR
    ├── text_util.py         # 한국어 띄어쓰기 교정 (kiwipiepy)
    └── mapping.py           # (예약)
```

---

## 4. 환경변수 (.env)

| 변수명               | 설명                             | 기본값                      |
| -------------------- | -------------------------------- | --------------------------- |
| `ASR_MODEL_PATH`     | Qwen3 ASR 모델 경로              | —                           |
| `ALIGNER_MODEL_PATH` | ForcedAligner 모델 경로          | —                           |
| `DIARIZE_MODEL_PATH` | pyannote 화자분리 모델 경로      | —                           |
| `HF_TOKEN`           | Hugging Face 액세스 토큰         | —                           |
| `OVERLAP_SEC`        | 슬라이딩 윈도우 오버랩 길이 (초) | `5.0`                       |
| `WINDOW_SEC`         | ASR 윈도우 크기 (초)             | `30.0`                      |
| `MYSQL_HOST`         | MySQL 호스트                     | `localhost`                 |
| `MYSQL_PORT`         | MySQL 포트                       | `3306`                      |
| `MYSQL_USER`         | MySQL 사용자                     | `root`                      |
| `MYSQL_PASSWORD`     | MySQL 비밀번호                   | `1234`                      |
| `MYSQL_DB`           | MySQL 데이터베이스명             | `meeting_assistant`         |
| `REDIS_HOST`         | Redis 호스트                     | `localhost`                 |
| `REDIS_PORT`         | Redis 포트                       | `6379`                      |
| `REDIS_DB`           | Redis DB 번호                    | `0`                         |
| `REDIS_TTL_SEC`      | Redis 키 TTL                     | `86400` (24h)               |
| `MONGO_URI`          | MongoDB 연결 URI                 | `mongodb://localhost:27017` |
| `MONGO_DB`           | MongoDB 데이터베이스명           | `meeting_assistant`         |

---

## 5. API 명세

### 5-1. REST

| 메서드 | 경로                    | 설명                    |
| ------ | ----------------------- | ----------------------- |
| `GET`  | `/health`               | 서버 헬스 체크          |
| `GET`  | `/meeting/{meeting_id}` | 회의 ID 확인 (테스트용) |

### 5-2. WebSocket

#### `WS /meeting/ws/stream/{meeting_id}`

실시간 음성 스트리밍 엔드포인트.

**연결 흐름**

```
1. WebSocket 연결 수립
2. 클라이언트 → 서버: JSON 전송  { "language": "Korean" }
3. 클라이언트 → 서버: 오디오 바이너리 반복 전송 (WebM / Opus / PCM 등)
4. 서버 → 클라이언트: JSON 응답 (매 청크마다)
5. 클라이언트 → 서버: 빈 바이트(b"") 전송 → 스트리밍 종료 신호
6. 서버: 최종 처리 후 완료 메시지 전송 후 연결 종료
```

> `diarization` 필드는 슬라이딩 윈도우 처리 후 화자분리 결과가 갱신될 때만 포함됨.

**서버 → 클라이언트 응답 스키마 (완료 시)**

```json
{
  "message": "Meeting processing complete",
  "meeting_id": "abc123"
}
```

---

## 6. 처리 파이프라인

### 6-1. 실시간 스트리밍 처리

```
오디오 청크 수신
    │
    ▼
bytes_to_wav16k()          ← WebM/Opus → WAV 16kHz 변환 (ffmpeg fallback 포함)
    │
    ▼
asr.streaming_transcribe() ← Qwen3 스트리밍 추론
    │
    ├─ [30초 누적 시] 슬라이딩 윈도우 처리
    │       ├── asr.finish_streaming_transcribe()  ← 현재 윈도우 확정
    │       ├── accumulated_text에 신규 텍스트 커밋
    │       ├── align_chunk()                      ← ForcedAligner 타임스탬프 정렬
    │       └── run_diarization()                  ← pyannote 화자분리 (슬라이딩)
    │
    ▼
Redis 업데이트
    ├── meeting:{id}:utterances  ← 화자별 발화 리스트 (JSON 배열)
    ├── meeting:{id}:speakers    ← spk_id → 사용자 이름 맵
    └── meeting:{id}:latest      ← 최신 인식 텍스트
    │
    ▼
WebSocket으로 클라이언트에 JSON 응답 전송
```

### 6-2. 회의 종료 후 오프라인 처리

```
빈 바이트 수신 (종료 신호)
    │
    ▼
offline_diarization()       ← 전체 오디오 화자분리 (최종 정확도 향상)
    │
    ▼
offline_asr_chunked()       ← 화자 세그먼트별 배치 ASR
    │
    ▼
build_minutes()             ← 회의록 포맷 생성 (seq, speaker, timestamp, text)
    │
    ▼
upload_mongodb()            ← MongoDB utterances 컬렉션에 저장
```

---

## 7. 데이터베이스

### MySQL (읽기 전용)

| 테이블                 | 역할                                        |
| ---------------------- | ------------------------------------------- |
| `meetings`             | 회의 정보 (workspace_id 등)                 |
| `meeting_participants` | 회의별 참가자 user_id                       |
| `users`                | 사용자 이름                                 |
| `speaker_profiles`     | 사용자별 음성 임베딩 (voice_embedding JSON) |

### Redis (실시간 캐시)

| 키                        | 타입   | 설명                  | TTL |
| ------------------------- | ------ | --------------------- | --- |
| `meeting:{id}:utterances` | List   | 화자별 발화 JSON 배열 | 24h |
| `meeting:{id}:speakers`   | Hash   | spk_id → 사용자 이름  | 24h |
| `meeting:{id}:latest`     | String | 최신 인식 텍스트      | 24h |

### MongoDB (최종 저장)

컬렉션: `utterances`

```json
{
  "meeting_id": "string",
  "workspace_id": "string",
  "created_at": "ISODate (KST)",
  "updated_at": "ISODate (KST)",
  "total_duration_sec": 1234,
  "meeting_start_time": "ISODate (KST)",
  "utterances": [
    {
      "seq": 1,
      "speaker_id": 1,
      "speaker_label": "홍길동",
      "timestamp": "2026-04-27T10:00:00",
      "text": "발화 내용"
    }
  ]
}
```

---

## 8. 모델 정보

| 모델                    | 역할                           | 비고                                              |
| ----------------------- | ------------------------------ | ------------------------------------------------- |
| Qwen3ASRModel           | 한국어 실시간/배치 음성 전사   | vLLM 기반, GPU 메모리 55% 사용                    |
| Qwen3ForcedAligner      | 단어 단위 타임스탬프 강제 정렬 | bfloat16, cuda:0                                  |
| pyannote.audio Pipeline | 화자분리 + 화자 임베딩 추출    | HF Token 필요, cosine similarity 0.5 이상 시 매칭 |

---

## 9. 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
cd asr_server
uvicorn main:app --host 0.0.0.0 --port 8000
```

CORS 허용 출처: `http://localhost:5173` (프론트엔드 개발 서버)

---

## 10. 타임존 정책

모든 시간 값은 **KST (UTC+9)** 기준으로 저장됩니다.  
`datetime.now(timezone.utc) + timedelta(hours=9)` 방식으로 처리.
