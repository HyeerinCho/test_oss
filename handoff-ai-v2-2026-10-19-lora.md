# Handoff: KTB4-17th AI V2 — 태깅·추출 LoRA/QLoRA 학습 준비 완료 (2026-10-19)

작성: 2026-10-19 · 클라우드 세션(맥 아님). 이전 인계: `handoff-ai-v2-2026-10-19.md` (얼굴 인증 등 열린 결정은 그 문서 그대로 유효).

## 사용자 / 작업 방식
- AI 파트 담당자(lena = Cho_hyerin, 커밋 작성자 `Cho_hyerin <rini0327@gmail.com>`). 한국어. **초보자용 단계별 + 표 정리**, 개념 축약 싫어함. 모바일로 자주 봄 → 결과 파일은 SendUserFile.
- Colab 노트북은 자기 노트북에 옮겨 씀 → 수정 시 **바뀐 셀 코드만** 채팅으로.
- 커밋 규칙: 팀 `gitconvention.md` (`feat: 제목` + 무엇을·왜, 브랜치 `feat/lena-*`). 푸시·PR 은 요청 시에만.
- `HyeerinCho/test_oss` 는 자유롭게 쓰는 개인 저장소 (`KTB-AI/`·`AI-FE/` 폴더에 팀 저장소 내용을 복사해 둠).

## 오늘 결정된 것
| 항목 | 결정 |
| --- | --- |
| 학습 방식 | 태깅·추출 **둘 다 LoRA/QLoRA**. 둘 다 학습해 품질 대비 비용이 나은 쪽 선택 |
| 순서 | **LoRA 먼저**(품질 기준점·더 빠름) → 같은 GPU 에서 QLoRA |
| 작업 위치 | 팀 저장소 `feat/lena-lora-qlora-compare` (최신 dev 기준). 기존 `training/` 파이프라인 재사용 |
| 추출 재라벨링 | **하지 않음** (점수 규칙은 같고 서술 규칙만 바뀜 — 아래 주의점) |
| 비교군 | TypeSafe **Jev** 를 태깅 비교군으로 추가하기로 함 (아직 미착수, API 키 필요) |
| 자원 판단 | 첫 버전을 Colab 에서 20스텝 시험 → 시간·비용 보고 Colab/RunPod 결정 |
| test_oss | `main`·`dev` 모두 팀 dev + 이번 작업 기준으로 맞춤 |

## 오늘 만든 것 — 팀 저장소 `100-hours-a-week/KTB4-17th-AI`, 브랜치 `feat/lena-lora-qlora-compare` (푸시됨, PR 없음)
커밋 `d0bbd97`, `11a8919` (dev `5906efd` 위).

| 파일 | 내용 |
| --- | --- |
| `training/tagging/train_lora.py` | 학습 후 `output-dir/run_metrics.json` (최대 GPU 메모리·시간·비용·eval loss·revision). `--gradient-checkpointing`, `--gpu-hourly-usd`, `--revision` 추가 |
| `training/extraction/from_teacher.py` (신규) | teacher 추출 라벨 → prompt/completion. familyId 분할, 허용 차원 밖 키 제거, 평가셋 답변 든 세션 제외, RUBRIC 변경 시 중단(`--allow-prompt-drift`) |
| `training/tagging/from_teacher.py` | 평가셋과 같은 답변을 미리 제외 (전에는 `build_sft.py` 가 1건 때문에 전체 중단) |
| `scripts/quality_metrics.py` + `run_langfuse_calibration.py` | Langfuse 실험에 태깅 primary/secondary F1·완전일치, 추출 차원 판정 정확도 (문항별 + 실행 전체) |
| `scripts/eval_tagging.py` | micro F1, **F2 게이트**(primary·off_topic 동시 일치 85%, 실패 5%) 판정 |
| `scripts/eval_extraction.py` (신규) | eval_tagging 과 같은 방식. **F3 게이트**(미답변 생략 100%, 범위 안 80%, 실패 5%). regression 18건 |
| `training/notebooks/lora_qlora_compare_colab.ipynb` | 설정 → 20스텝 시험(전체 시간·비용 추정) → LoRA → QLoRA → 비교표 |
| `training/RESULTS.md` | 결과 표 틀 (모델별·게이트·학습 비용·버전별·보는 법·갱신하는 법). 사용자가 다른 프로젝트에서 쓰던 형식 따름 |
| `training/README.md` | 비교·추출 학습 방법, 로컬 RTX 3060 6GB 시험법, 재라벨링 명령 |

검증: 전체 테스트 1,947 통과, ruff 통과. 팀 고정 버전(transformers 5.19, trl 1.14)으로 작은 모델 CPU 학습·20스텝 셀 확인.
**GPU 실학습·QLoRA(4비트)는 아직 한 번도 안 돌림.**

## 학습 데이터 (사용자에게 `sft-pilot200.zip` 으로 전달)
- 원천: 맥 `data/teacher/pilot200` (qwen/qwen3.8-27b, 합성 세션 200개). 사용자가 zip 으로 올려 줌.
- 태깅 train 1,696 / val 198 (일치율 0.6 미만 172건·누수 1건 제외), 추출 train 178 / val 20 (스키마 오류 1·누수 세션 1 제외).
- 최대 길이: 태깅 975, 추출 3,257 토큰 → 노트북 `--max-length` 1280 / 4096.
- Drive 에 `ktb-ai/sft/{tagging,extraction}/out/` 로 올리면 노트북 기본 경로.
- ⚠️ 그 전에 준 `teacher-train-data.zip` 은 평가셋 누수 1건("몰라.") 포함 → **쓰지 말 것**.
- 검수 전 teacher 라벨이라 `--allow-unreviewed`(시험용). 팀 `build_sft.py` 는 원래 검수 전 라벨을 막는다.

## 꼭 알아야 할 발견
1. **교사가 운영 모델보다 태깅을 못한다** (골드셋 60건, 현 TAG_PROMPT): Gemini 2.5 flash-lite primary F1 **90.2** / 완전일치 86.7 vs 교사 Qwen **83.1** / 71.7.
   교사 주 오류는 과잉 태깅(`disclosure`·`assurances` 덧붙임). 학생은 교사 수준까지 → 운영보다 낮을 가능성. 추출은 교사가 더 나음(차원 판정 100%, 범위 24/24).
   → **미결정:** 목표를 "운영보다 싸고 빠르며 F2 게이트 85% 통과"로 둘지, 교사 개선(추론 켜기·프롬프트)·교체할지.
2. **지금 평가는 greedy 가 아니다.** 앱은 temperature 를 안 보내고 `serving/vllm/config.yaml` 은 `generation-config: vllm`(= temperature 1.0). 평가 서버는 `--override-generation-config '{"temperature": 0}'` 로 띄운다 (운영 코드는 안 바꿈).
3. **평가셋이 작다:** 태깅 regression 47건(1건 ≈ 2.1%p, 95% CI 약 ±13%p), 추출 18건. "1%p 이내면 싼 쪽" 기준은 폐기 → "오차 범위 안이면 같은 수준".
4. **프롬프트 버전:** pilot200 태깅 라벨 `67757f18` = 팀 dev 와 같음 ✅. 추출 라벨 RUBRIC `0481859d` ≠ dev `9749c6f4` — 차이는 서술(body)·traits 규칙뿐(7e10dc0: 답변 인용 → 성향 요약, 비속어 중립화). 점수 영향 없음, **서술은 옛 규칙(인용)을 배움**. 재라벨링은 추출만 약 $0.5 이하 추정:
   `cp pilot200/tagging.jsonl pilot200-rubric2/ && label_with_teacher.py --synthetic data/synthetic/pilot200.jsonl --out-dir data/teacher/pilot200-rubric2 --resume`
5. T4 는 BF16 미지원 → `train_lora.py` 가 원본을 FP32 로 올려 LoRA OOM. **L4·A100 필요.**
6. `config.yaml` 은 `max-loras: 1` → 어댑터 여러 개 평가 시 `--max-loras 4` 와 `--lora-modules` 를 함께 준다 (RESULTS.md "갱신하는 법").
7. 팀 `build_sft.py` 는 문항 id 로 train/val 분할 → 같은 세션이 양쪽에 섞여 eval loss 낙관적 (LoRA·QLoRA 동일 영향, 미수정).

## 질문에 대한 답 (오늘)
- **RTX 3060 노트북 6GB + RAM 32GB:** LoRA 불가(BF16 가중치만 약 6.4GB). QLoRA 태깅은 배치 1·gradient checkpointing 으로 빠듯하게 가능(추정), 추출은 어려움. 시험·디버깅용. WSL2 권장. README 의 20스텝 명령으로 실측.
- **Colab vs RunPod:** 노트북 4번 셀(20스텝 시험)로 결정. 한 번 2시간 안쪽이면 Colab, 3시간 넘거나 여러 번 돌리면 RunPod. 서빙·평가(vLLM)는 어차피 RunPod.
- **Jev(TypeSafe):** 결정 전용 닫힌 API(파인튜닝 불가, 영어 우선). 태깅은 차원 19개 × `choice`(primary/secondary/없음) + off_topic `noul` 로 한 요청에 묶어 비교 가능. API: `POST https://api.typesafe.ai/v1/systemone`, $0.042/M 입력 토큰.

## 내일 할 일 (사용자)
1. `sft-pilot200.zip` 풀어 Drive `ktb-ai/` 아래 `sft/` 업로드
2. Colab 에서 `training/notebooks/lora_qlora_compare_colab.ipynb` (팀 저장소 브랜치를 클론함), **L4/A100** 선택, `GPU_HOURLY_USD` 입력
3. 4번 셀(20스텝 시험) 결과로 Colab/RunPod 결정 → 5번 LoRA → 6번 QLoRA → 7번 비교표
4. 결과(특히 4번 셀 출력)를 다음 세션에 붙여 주기

## 다음 세션 할 일 제안
1. 시험 실행 결과로 Colab/RunPod 판단, 학습 결과를 `training/RESULTS.md` 에 채우기
2. RunPod vLLM(greedy)로 Gemini·Kanana 학습 전·LoRA·QLoRA 평가 → 게이트 판정
3. 교사 문제(발견 1) 방향 결정
4. Jev 비교군 연결 (API 키 받으면)
5. 필요 시 `feat/lena-lora-qlora-compare` → dev PR

## 환경 / git 상태
| 저장소 | 브랜치 | 상태 |
| --- | --- | --- |
| 팀 `100-hours-a-week/KTB4-17th-AI` | `feat/lena-lora-qlora-compare` | `11a8919` 푸시됨, PR 없음 |
| 개인 `HyeerinCho/test_oss` | `main` | `944f270` — KTB-AI = 팀 dev(5906efd) + 위 브랜치(11a8919), AI-FE = FE dev(ec2e509) |
| 개인 `HyeerinCho/test_oss` | `dev` | `854023a` — main 과 내용 동일 (기존 기록 위에 커밋) |
| 개인 `HyeerinCho/test_oss` | `claude/funny-fermat-t56jip` | `944f270` (main 과 같음) |

- test_oss 는 팀 저장소를 자동으로 따라가지 않는다. 다시 맞출 때는 `git archive` 로 KTB-AI 교체 (한글 파일명은 `core.quotepath=false`, `docs/` 가 gitignore 라 `git add -f --pathspec-from-file`).
- 맥에만 있던 teacher 스크립트 5개(`label_with_teacher.py` 등)는 팀 dev 에 이미 있음.
- 맥 작업 트리(`fix/lena-llm-timeout-retry`)의 `docs/` 산출물 갱신(V2 문서 2종 학습 범위 반영)은 아직 안 함.

## Suggested skills
- `diagnosing-bugs` — Colab·bitsandbytes·vLLM 실행 오류
- `tdd` — 평가 스크립트·변환기 수정 시 (테스트: `tests/test_eval_extraction.py`, `test_extraction_from_teacher.py`, `test_quality_metrics.py`, `test_tagging_finetune.py`)
- `team-git-convention` — 커밋·PR
