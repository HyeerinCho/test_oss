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
| 교사 | **Qwen 유지 + 태깅 라벨 만장일치(5/5)** (= 안 A). Gemini 교사는 약관 확인 전까지 보류 |
| 비교군 | TypeSafe **Jev** 를 태깅 비교군으로 추가하기로 함 (아직 미착수, API 키 필요) |
| 자원 판단 | 첫 버전을 Colab 에서 20스텝 시험 → 시간·비용 보고 Colab/RunPod 결정 |
| test_oss | `main`·`dev` 모두 팀 dev + 이번 작업 기준으로 맞춤 |

## 오늘 만든 것 — 팀 저장소 `100-hours-a-week/KTB4-17th-AI`, 브랜치 `feat/lena-lora-qlora-compare` (푸시됨, PR 없음)
커밋 `d0bbd97`, `11a8919`, `217e119` (dev `5906efd` 위).

| 파일 | 내용 |
| --- | --- |
| `training/tagging/train_lora.py` | 학습 후 `output-dir/run_metrics.json` (최대 GPU 메모리·시간·비용·eval loss·revision). `--gradient-checkpointing`, `--gpu-hourly-usd`, `--revision` 추가 |
| `training/extraction/from_teacher.py` (신규) | teacher 추출 라벨 → prompt/completion. familyId 분할, 허용 차원 밖 키 제거, 평가셋 답변 든 세션 제외, RUBRIC 변경 시 중단(`--allow-prompt-drift`) |
| `training/tagging/from_teacher.py` | 평가셋과 같은 답변을 미리 제외 (전에는 `build_sft.py` 가 1건 때문에 전체 중단). `--min-vote-share` (1.0 = 만장일치, 기본은 다수결) |
| `scripts/quality_metrics.py` + `run_langfuse_calibration.py` | Langfuse 실험에 태깅 primary/secondary F1·완전일치, 추출 차원 판정 정확도 (문항별 + 실행 전체) |
| `scripts/eval_tagging.py` | micro F1, **F2 게이트**(primary·off_topic 동시 일치 85%, 실패 5%) 판정 |
| `scripts/eval_extraction.py` (신규) | eval_tagging 과 같은 방식. **F3 게이트**(미답변 생략 100%, 범위 안 80%, 실패 5%). regression 18건 |
| `training/notebooks/lora_qlora_compare_colab.ipynb` | 설정 → 20스텝 시험(전체 시간·비용 추정) → LoRA → QLoRA → 비교표 |
| `training/RESULTS.md` | 결과 표 틀 (모델별·게이트·학습 비용·버전별·보는 법·갱신하는 법). 사용자가 다른 프로젝트에서 쓰던 형식 따름 |
| `training/README.md` | 비교·추출 학습 방법, 로컬 RTX 3060 6GB 시험법, 재라벨링 명령 |

검증: 전체 테스트 1,947 통과, ruff 통과. 팀 고정 버전(transformers 5.19, trl 1.14)으로 작은 모델 CPU 학습·20스텝 셀 확인.
**GPU 실학습·QLoRA(4비트)는 아직 한 번도 안 돌림.**

## 학습 데이터 (사용자에게 **`sft-pilot200-v1.zip`** 으로 전달 — 태깅 만장일치)
- 원천: 맥 `data/teacher/pilot200` (qwen/qwen3.8-27b, 합성 세션 200개). 사용자가 zip 으로 올려 줌.
- 태깅 train 1,696 / val 198 (일치율 0.6 미만 172건·누수 1건 제외), 추출 train 178 / val 20 (스키마 오류 1·누수 세션 1 제외).
- 태깅 정답은 `from_teacher.py --min-vote-share 1.0`. 다수결 대비 primary 라벨 2,055 → 1,741, primary 가 바뀐 행 295 (그중 빈 primary 가 된 행 115).
- 최대 길이: 태깅 975, 추출 3,257 토큰 → 노트북 `--max-length` 1280 / 4096.
- Drive 에 `ktb-ai/sft/{tagging,extraction}/out/` 로 올리면 노트북 기본 경로.
- ⚠️ 그 전에 준 `teacher-train-data.zip`(누수 1건 "몰라." 포함)과 `sft-pilot200.zip`(다수결 라벨)은 **쓰지 말 것**.
- 검수 전 teacher 라벨이라 `--allow-unreviewed`(시험용). 팀 `build_sft.py` 는 원래 검수 전 라벨을 막는다.

## 꼭 알아야 할 발견
1. **교사가 운영 모델보다 태깅을 못했다** (골드셋 60건, 현 TAG_PROMPT): Gemini 2.5 flash-lite primary F1 **90.2** / 완전일치 86.7 vs 교사 Qwen **83.1** / 71.7.
   교사 주 오류는 과잉 태깅. 추출은 교사가 더 나음(차원 판정 100%, 범위 24/24).
   → **해결(안 A):** 이미 있는 5회 샘플 득표로 만장일치만 남기니 (골드셋 63건, eval_tagging 채점)

   | 규칙 | primary F1 | 완전 일치 | F2 게이트 | 덧붙임 | 놓침 |
   | --- | --- | --- | --- | --- | --- |
   | Qwen 3/5 (기존) | 83.1 | 69.8 | 73.0 | 15 | 0 |
   | Qwen 4/5 | 87.1 | 77.8 | 77.8 | 11 | 0 |
   | **Qwen 5/5 (채택)** | **89.2** | **81.0** | 81.0 | 9 | 0 |
   | Gemini 2.5 flash-lite 3/5 | 90.2 | 77.8 | 87.3 | 8 | 0 |
   | Gemini 2.5 flash-lite 4/5 | 91.1 | 81.0 | 87.3 | 6 | 1 |

   - 규칙을 골드셋을 보고 골랐으므로 낙관적 → 최종 판정은 blind_holdout.
   - 남는 오류 12건: 간접 단서 과잉 태깅 9, 욕설("시발", "꺼져") off_topic 놓침 2, 농담을 off_topic 으로 1. 골드 라벨도 검수 전이라 일부는 라벨 쪽이 애매.
   - **Gemini 교사 보류 이유:** Gemini API 추가 약관(2024-05~2025-11 보관본 모두)이 "서비스와 경쟁하는 모델 개발"에 사용 금지. Kanana 가 운영의 Gemini 호출을 대체하는 목적이라 걸릴 여지 → 팀·멘토 확인 필요. OpenRouter 경유도 같음. 확인되면 `TEACHER_LLM_MODEL=google/gemini-2.5-flash-lite` 로 pilot200 재라벨링(약 $1~2 추정), 태깅만 Gemini·추출은 Qwen 조합도 가능.
2. **지금 평가는 greedy 가 아니다.** 앱은 temperature 를 안 보내고 `serving/vllm/config.yaml` 은 `generation-config: vllm`(= temperature 1.0). 평가 서버는 `--override-generation-config '{"temperature": 0}'` 로 띄운다 (운영 코드는 안 바꿈).
3. **평가셋이 작다:** 태깅 regression 47건(1건 ≈ 2.1%p, 95% CI 약 ±13%p), 추출 18건. "1%p 이내면 싼 쪽" 기준은 폐기 → "오차 범위 안이면 같은 수준".
4. **프롬프트 버전:** pilot200 태깅 라벨 `67757f18` = 팀 dev 와 같음 ✅. 추출 라벨 RUBRIC `0481859d` ≠ dev `9749c6f4` — 차이는 서술(body)·traits 규칙뿐(7e10dc0: 답변 인용 → 성향 요약, 비속어 중립화). 점수 영향 없음, **서술은 옛 규칙(인용)을 배움**. 재라벨링은 추출만 약 $0.5 이하 추정:
   `cp pilot200/tagging.jsonl pilot200-rubric2/ && label_with_teacher.py --synthetic data/synthetic/pilot200.jsonl --out-dir data/teacher/pilot200-rubric2 --resume`
5. T4 는 BF16 미지원 → `train_lora.py` 가 원본을 FP32 로 올려 LoRA OOM. **L4·A100 필요.**
6. `config.yaml` 은 `max-loras: 1` → 어댑터 여러 개 평가 시 `--max-loras 4` 와 `--lora-modules` 를 함께 준다 (RESULTS.md "갱신하는 법").
7. **학습 데이터의 차원 쏠림:** 만장일치 train 기준 primary 예시 — 감정적 맞대응 2, 관계 확신 표현 3, 피하고 싶은 것 3, 솔직한 관계 대화 5, 매력·활발함 8, 긍정적 상호작용 10 (6개 차원 15건 미만). 합성기 질문이 그 차원을 거의 안 물어서 생김 → 세션 수만 늘려선 안 채워짐. 평가 regression 에도 7개 차원은 정답이 0건.
8. 팀 `build_sft.py` 는 문항 id 로 train/val 분할 → 같은 세션이 양쪽에 섞여 eval loss 낙관적 (LoRA·QLoRA 동일 영향, 미수정).

## 데이터셋 구조와 필요한 양 — 그림: [태깅·추출 데이터셋 지도](https://claude.ai/artifact/VN1TdaVYtXN6tEjPETUynY) (비공개 artifact)
| 데이터 | 태깅 | 추출 | 메모 |
| --- | --- | --- | --- |
| 평가셋 (Langfuse v1.13, 사람 검수 전) | 80 = calibration 16 · regression 47 · holdout 16 (+장애 주입 1) | 60 = 채점 30 (6·18·6) + code_behavior 30 | off_topic 정답 regression 4건 |
| 골드셋 (= 평가셋 calibration + regression) | 63 | 24 | 교사 채점용. 별도 세트 아님 |
| 사람 골드셋 `human/` | 0 | 0 | 폴더만 있음 |
| 학습셋 (pilot200 교사 라벨) | train 1,696 · val 198 | train 178 · val 20 | |

- 오차(정답률 80% 근처, 95%): 47건 ±11.4%p · 63건 ±9.9 · 100건 ±7.8 · 200건 ±5.5 · 400건 ±3.9 · 683건 ±3.0.
- 제안(통계 + 경험칙): 태깅 평가 200건 이상(차원마다 정답 5건 이상, off_topic 20건 이상), 추출 평가 60~100건, 사람 골드셋 태깅 100·추출 30,
  태깅 학습 차원마다 100건 이상(총 3,000~5,000, 드문 차원을 겨누는 질문·시드 추가), off_topic 학습 200건 이상(지금 57), 추출 학습 1,000세션(약 $42).

## 질문에 대한 답 (오늘)
- **RTX 3060 노트북 6GB + RAM 32GB:** LoRA 불가(BF16 가중치만 약 6.4GB). QLoRA 태깅은 배치 1·gradient checkpointing 으로 빠듯하게 가능(추정), 추출은 어려움. 시험·디버깅용. WSL2 권장. README 의 20스텝 명령으로 실측.
- **Colab vs RunPod:** 노트북 4번 셀(20스텝 시험)로 결정. 한 번 2시간 안쪽이면 Colab, 3시간 넘거나 여러 번 돌리면 RunPod. 서빙·평가(vLLM)는 어차피 RunPod.
- **Jev(TypeSafe):** 결정 전용 닫힌 API(파인튜닝 불가, 영어 우선). 태깅은 차원 19개 × `choice`(primary/secondary/없음) + off_topic `noul` 로 한 요청에 묶어 비교 가능. API: `POST https://api.typesafe.ai/v1/systemone`, $0.042/M 입력 토큰.

## 내일 할 일 (사용자)
1. **`sft-pilot200-v1.zip`** 풀어 Drive `ktb-ai/` 아래 `sft/` 업로드
2. Colab 에서 `training/notebooks/lora_qlora_compare_colab.ipynb` (팀 저장소 브랜치를 클론함), **L4/A100** 선택, `GPU_HOURLY_USD` 입력
3. 4번 셀(20스텝 시험) 결과로 Colab/RunPod 결정 → 5번 LoRA → 6번 QLoRA → 7번 비교표
4. 결과(특히 4번 셀 출력)를 다음 세션에 붙여 주기

## 다음 세션 할 일 제안
1. 시험 실행 결과로 Colab/RunPod 판단, 학습 결과를 `training/RESULTS.md` 에 채우기
2. RunPod vLLM(greedy)로 Gemini·Kanana 학습 전·LoRA·QLoRA 평가 → 게이트 판정
3. Gemini 교사 약관 확인 결과에 따라 재라벨링 여부 결정
4. 데이터 늘리기: 드문 차원을 겨누는 합성 질문·시드 추가, 평가셋 확대와 사람 검수 계획
5. Jev 비교군 연결 (API 키 받으면)
6. 필요 시 `feat/lena-lora-qlora-compare` → dev PR

## 환경 / git 상태
| 저장소 | 브랜치 | 상태 |
| --- | --- | --- |
| 팀 `100-hours-a-week/KTB4-17th-AI` | `feat/lena-lora-qlora-compare` | `217e119` 푸시됨, PR 없음 |
| 개인 `HyeerinCho/test_oss` | `main` | `944f270` — KTB-AI = 팀 dev(5906efd) + 위 브랜치(11a8919), AI-FE = FE dev(ec2e509) |
| 개인 `HyeerinCho/test_oss` | `dev` | `854023a` — main 과 내용 동일 (기존 기록 위에 커밋) |
| 개인 `HyeerinCho/test_oss` | `claude/funny-fermat-t56jip` | main + 이 인계 문서 커밋 (main·dev 에는 인계 문서 없음) |

- test_oss main·dev 의 KTB-AI 는 팀 브랜치 `11a8919` 기준이라 `217e119`(만장일치 옵션)는 아직 안 들어가 있다.

- test_oss 는 팀 저장소를 자동으로 따라가지 않는다. 다시 맞출 때는 `git archive` 로 KTB-AI 교체 (한글 파일명은 `core.quotepath=false`, `docs/` 가 gitignore 라 `git add -f --pathspec-from-file`).
- 맥에만 있던 teacher 스크립트 5개(`label_with_teacher.py` 등)는 팀 dev 에 이미 있음.
- 맥 작업 트리(`fix/lena-llm-timeout-retry`)의 `docs/` 산출물 갱신(V2 문서 2종 학습 범위 반영)은 아직 안 함.

## Suggested skills
- `diagnosing-bugs` — Colab·bitsandbytes·vLLM 실행 오류
- `tdd` — 평가 스크립트·변환기 수정 시 (테스트: `tests/test_eval_extraction.py`, `test_extraction_from_teacher.py`, `test_quality_metrics.py`, `test_tagging_finetune.py`)
- `team-git-convention` — 커밋·PR
