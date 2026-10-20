# test_oss 변경 이력 및 인수인계

다른 세션에서 이어서 작업하기 위한 문서다. 2026-10-19 ~ 2026-10-20 에 이 저장소에서 바뀐 내용, 현재 상태, 남은 작업을 정리한다.

## 1. 현재 저장소 구조

원래 `README.md`, `main.c` 만 있던 test_oss 를 비우고, KTB4 팀 프로젝트 두 개를 폴더별로 가져왔다. 파일만 복사했으며 원본 저장소의 커밋 히스토리는 없다.

```
test_oss/
├── .github/workflows/
│   ├── ai-ci.yml          # KTB-AI CI (working-directory: KTB-AI)
│   └── frontend-ci.yml    # AI-FE CI  (working-directory: AI-FE)
├── KTB-AI/                # https://github.com/100-hours-a-week/KTB4-17th-AI
└── AI-FE/                 # https://github.com/100-hours-a-week/KTB4-17th-FE
```

| test_oss 브랜치 | 커밋 | `KTB-AI/` 기준 | `AI-FE/` 기준 |
| --- | --- | --- | --- |
| `main` | `944f270` | AI `dev`(5906efd) + `feat/lena-lora-qlora-compare`(11a8919) | FE `dev`(ec2e509) |
| `dev` | `a6c85d7` 이후 | `main` 과 같음 + WebM 라이브니스 수정(3절) | FE `dev`(ec2e509) |

- `main` 과 `dev` 의 차이는 3절의 WebM 수정(파일 2개)뿐이다.
- `main` 의 `KTB-AI/` 는 처음에 AI `main`(aafeb44)이었으나, 다른 세션(`session_01FaFbjRG1b2Sy8ENzK2EPCJ`)이 위 기준으로 교체하고 `dev` 도 같은 내용으로 맞췄다(`854023a`).

## 2. 변경 이력

| 날짜 | 브랜치 | 커밋 | 내용 |
| --- | --- | --- | --- |
| 10-19 | dev | `5dd2af8` | 기존 파일 삭제, KTB4-17th-AI `dev`(4f9860c) 스냅샷으로 교체 |
| 10-19 | main | `5c3e2be` | 기존 파일 삭제, KTB4-17th-AI `main`(aafeb44) 스냅샷으로 교체 |
| 10-19 | — | — | 작업 브랜치 `claude/affectionate-mendel-xg5nj8` 를 `dev` 로 이름 변경 |
| 10-19 | dev | `d44b9a7` | 루트 파일을 `KTB-AI/` 로 이동, CD 워크플로 삭제, 누락 문서 31개 복원 |
| 10-19 | main | `b636e9e` | 같은 작업(누락 문서 33개 복원) |
| 10-19 | dev | `22d08f3` | KTB4-17th-FE `dev`(ec2e509)를 `AI-FE/` 로 추가, `frontend-ci.yml` 추가 |
| 10-19 | main | `fb708bd` → `a8a6368` | FE `main` 을 넣었다가 요청에 따라 FE `dev`(ec2e509)로 교체 |
| 10-19 | main | `944f270` 등 | (다른 세션) KTB-AI 를 AI `dev`(5906efd) + LoRA/QLoRA 비교 브랜치로 교체 |
| 10-19 | dev | `854023a` | (다른 세션) `dev` 를 `main`(944f270)과 같은 내용으로 맞춤 |
| 10-20 | dev | `a6c85d7` | WebM 영상 라이브니스 프레임 샘플링 수정(3절) |

### CI/CD 정리

- 삭제: `dev-cd.yml`(ECR/EC2 DEV 배포), `release.yml`(semantic-release), `prod-cd.yml`(운영 배포), 두 저장소의 `discord-pr-review-reminder.yml`, FE 의 `frontend-cd.yml`·`frontend-dev-cd.yml`
- 유지: `ai-ci.yml`, `frontend-ci.yml`. 둘 다 `dev` 로 올리는 PR 에서 실행된다.
  - `ai-ci.yml`: `defaults.run.working-directory: KTB-AI`, `cache-dependency-glob: KTB-AI/uv.lock` 추가
  - `frontend-ci.yml`: `working-directory: AI-FE`, `cache-dependency-path: AI-FE/package-lock.json`, CodeQL `source-root: AI-FE` 추가
- 삭제 전에 실행된 기록: `AI Release` 성공(chore 커밋이라 태그·배포 없음), `AI DEV CD` 는 AWS 시크릿이 없어 실패(배포 안 됨)

### 가져오기 때 주의한 점

원본 저장소에는 `.gitignore` 에 걸리지만 커밋된 파일이 있다(`docs/v2docs/openapi.json`, ADR 문서, `.claude/launch.json` 등). `git add -A` 로 복사하면 이 파일들이 빠진다. 실제로 처음에 빠져서 `test_profile_trust_openapi` 가 실패했고, 원본의 `git ls-files` 목록으로 `git add -f` 하여 복원했다. 다시 가져올 때도 원본 `git ls-files` 기준으로 복사해야 한다.

## 3. WebM 라이브니스 수정 (`a6c85d7`, dev 에만 있음)

- 파일: `KTB-AI/app/features/face_verification/liveness.py`, `KTB-AI/tests/test_face_verification_challenge.py`
- 문제: 브라우저 MediaRecorder 로 녹화한 WebM 은 길이(Duration) 정보가 없다. 이 경우 ffmpeg 이 길이를 INT64_MIN 으로 주고, OpenCV 는 이 값에 시간 단위(1/1000초)와 fps(24)를 곱해 프레임 수를 −2.2136e17 로 읽는다. 기존 코드는 이때 앞 24프레임(0~0.96초)만 분석했다.
- 수정: 프레임 수가 1 이하이면 `_count_frames()` 로 한 번 훑어 실제 프레임 수를 세고, 영상 전체에서 24장을 고르게 뽑는다(5초 영상 기준 약 30ms 추가).
- 결과: 브라우저 녹화식 5초 WebM 의 분석 구간이 0~0.96초에서 0~4.96초로 넓어진다. mp4(Safari, 최신 Chrome)는 원래 문제가 없었고 동작도 그대로다.
- 검증: 새 테스트 `test_sampling_spans_whole_video_when_frame_count_is_missing` 는 수정 전 코드에서 실패하고 수정 후 통과한다. 전체 테스트 1948개, ruff check/format 통과.

## 4. 얼굴 인증 FE ↔ AI 연결 요약

FE 는 AI 를 직접 부르지 않고 메인 백엔드(이 저장소에 없음)를 거친다. 계약 문서는 `KTB-AI/docs/v2docs/backend-api-contract.md` 다.

| FE (`AI-FE/src/features/profiletrust/api.js`) | 백엔드가 부르는 AI API (`/ai/api/v1/profile-trust/...`) |
| --- | --- |
| `POST /api/v1/files/upload-intents/{id}/primary-photo-check` | `photos/primary/frontal-check`, `photos/primary/synthetic-check` |
| `POST /api/v1/users/me/face-verifications` | `verifications/challenges` |
| `POST /api/v1/users/me/face-verifications/{id}/attempts?attemptNo=N` | `verifications/complete` |
| `GET /api/v1/users/me/face-verifications/{id}` | 없음(백엔드 상태) |

- 판정·이유 코드(`VERIFIED/RETRY/NOT_VERIFIED`, `LIVENESS_FAILED/FACE_MISMATCH/FACE_COMPARISON_FAILED/NO_FACE`)와 대표사진 이유 코드는 FE `model.js` 와 AI 가 일치한다.
- FE 녹화: 640×480, 24fps, 1Mbps, 5초(`CLIP_DURATION_MS`), mp4 우선이고 안 되면 webm.
- AI 모델: 얼굴 검출 YuNet, 얼굴 특징점 LBF(고개 각도 계산), 동일인 비교 SFace(`FACE_MATCH_THRESHOLD` 0.42, 후보 프레임 유사도 중앙값). 모델 파일은 `python scripts/download_models.py` 로 받는다.

## 5. 남은 작업

### 바로 할 수 있는 것

- [ ] **WebM 수정을 `main` 에 반영할지 결정**: 지금은 `dev` 에만 있다.
- [ ] **계약 문서 녹화 시간 수정**: `backend-api-contract.md` 는 "4초 영상"이라고 되어 있지만 FE 는 5초를 녹화한다.
- [ ] **옛 브랜치 삭제**: 원격 `claude/affectionate-mendel-xg5nj8`(`dev` 의 옛 이름, 내용은 `5dd2af8`)은 세션 프록시가 브랜치 삭제를 막아(HTTP 403) 남아 있다. GitHub 웹 화면에서 지워야 한다. 다른 세션이 만든 `claude/funny-fermat-t56jip` 도 남아 있다.
- [ ] (선택) CI 경로 조건: 지금은 `dev` 로 올리는 모든 PR 에서 두 CI 가 함께 돈다. `paths: [KTB-AI/**]`, `paths: [AI-FE/**]` 로 나눌 수 있다.

### 다음 기능: 동작 지시형 라이브 인증 (설계 단계, 미착수)

목표: 정면 응시만 하던 라이브 인증을 "고개를 좌우·상하로 살짝 움직이기 + 손 모양 따라 하기"로 바꾼다. 무작위 동작으로 미리 찍어둔 영상이나 사진 재생을 막는다.

지금 구조에서 그대로 쓸 수 있는 것:

- `ChallengeSigner` 가 챌린지 목록을 HMAC 토큰에 서명해 넣는다. 서버가 무작위로 고르기만 하면 된다.
- `FaceAnalyzer.observe()` 가 프레임마다 yaw/pitch/roll 을 계산한다(`primary_photo/analyzer.py` `_estimate_pose`, LBF 기반의 대략적인 값).
- 현재 `Challenge` 는 `LOOK_STRAIGHT` 하나뿐이고, `_completed_challenges()` 는 그 외 목록이면 빈 리스트를 반환한다(`liveness.py`).

권장 순서:

1. **고개 동작 (AI, 새 모델 없음)**
   - `Challenge` 에 `TURN_LEFT`, `TURN_RIGHT`, `LOOK_UP`, `LOOK_DOWN` 을 추가하고, `ChallengeSigner.create()` 가 무작위로 고르게 한다.
   - 시간 순서로 "정면 → 동작 → 정면"을 판정한다.
   - 동일인 비교는 정면 프레임으로만 한다.
   - 테스트를 추가한다.
2. **손 모양 (AI, 새 모델)**
   - MediaPipe Hand Landmarker(Apache-2.0, CPU, 손가락 관절 21개 점)를 추가하고, 그 점들로 손 모양을 규칙 판정한다.
   - 후보: `HAND_V`, `HAND_PALM`, `HAND_THUMBS_UP`, `HAND_ON_CHEST`(첨부 예시처럼 손을 가슴에 대기).
   - `scripts/download_models.py`, `models/README.md`(라이선스), `pyproject.toml` 을 갱신한다.
3. **프레임 샘플링**: 동작 여러 개를 확인하려면 5초·24장으로는 부족하다. 녹화 7~8초, 분석 약 40장, 또는 단계별 판정을 검토한다.
4. **FE**: 지금 FE 는 챌린지 목록을 받지 않는다(`captureHint` 는 정면 안내만 한다). 단계별 예시 이미지, 안내 문구, 진행률을 표시하는 화면이 필요하다.
5. **백엔드 계약**: AI 의 `challenges` 를 세션 응답으로 FE 에 전달하도록 `backend-api-contract.md` 를 갱신한다.

결정이 필요한 것(사용자 확인 전):

- 한 번 인증할 때 시킬 동작 개수(추천: 고개 1개 + 손 1개)
- 사용할 손 모양 종류
- 좌우 기준: 전면 카메라 미리보기는 거울 반전되지만 녹화 파일은 반전되지 않는 경우가 많다. 사용자 기준과 화면 기준 중 무엇으로 할지 정하고 실제 기기로 확인해야 한다.

위험 요소:

- LBF 로 계산한 각도는 대략적인 값이다. "살짝"(15~20°)으로 판정 기준을 잡고 실제 데이터로 조정해야 한다.
- 손이 얼굴을 가리면 동일인 비교가 실패한다. 손 동작은 턱 아래에서 하도록 안내하고, 비교에서는 손이 없는 프레임만 쓴다.
- 손을 쓰기 어려운 사용자를 위한 대체 경로(고개 동작만)가 필요하다.
- 실시간 딥페이크는 동작을 따라 할 수 있다. 재생 공격을 막는 장치이지 완전한 차단은 아니다.
- ADR 0004(인증 후 생체 데이터 미보관), ADR 0006(얼굴 데이터 국내 처리)을 지켜야 한다. 손 인식 모델도 서버에서 직접 돌린다.

## 6. 로컬 검증 방법

```bash
# AI (CI 와 같은 순서)
cd KTB-AI
uv sync --all-groups --locked
uv run ruff check . && uv run ruff format --check .
uv run pytest --ignore-glob='tests/test_*_api.py'
uv run pytest tests/test_*_api.py

# FE
cd AI-FE
npm ci
npm run lint:biome && npm run lint:html && npm run lint:css
node --test "src/**/*.test.js"
npm run build
```

브라우저 녹화식 WebM 은 ffmpeg 으로 재현할 수 있다:

```bash
ffmpeg -f lavfi -i testsrc=size=640x480:rate=24 -t 5 -c:v libvpx -b:v 1M -f webm -live 1 live.webm
```

## 7. 세션 환경 메모

- 이 클라우드 세션의 git 프록시는 원격 브랜치 삭제를 막는다(403). 브랜치 삭제는 GitHub 웹에서 한다.
- 여러 세션이 같은 `main`·`dev` 에 푸시한다. 푸시 전에 `git fetch` 로 원격이 바뀌었는지 확인한다.
