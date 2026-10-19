"""teacher 라벨 → student(LoRA/QLoRA) 학습 데이터 — label_with_teacher.py 출력을 Colab 노트북 형식으로 바꾼다.

한 줄 = {"messages": [system, user, assistant]}. system·user 는 운영 코드(TAG_PROMPT·RUBRIC·입력 형식)와
글자 하나까지 같고, assistant 는 teacher 라벨(JSON 문자열)이다.

  거르는 것
    - 골드셋(source=gold): Langfuse 평가셋과 같은 문항이므로 학습에 넣으면 평가가 무의미해진다
    - 실패·파싱 오류·스키마 오류가 난 라벨
    - 태깅 일치율이 --min-agreement 미만인 답 (teacher 도 헷갈린 답, check_synthetic_labels 의 LOW_AGREEMENT)
    - 라벨을 만든 뒤 운영 프롬프트가 바뀐 경우 전체 중단 (--allow-prompt-drift 로 무시)
  정리하는 것
    - 태깅 정답 = 다수결(majority). off_topic 이면 primary·secondary 를 비운다 (label_with_teacher 와 같음)
    - 추출 정답 = teacher 원출력에서 허용 차원(answered) 밖 점수·텍스트 키와 스키마에 없는 키를 지운 것.
      운영 후처리가 어차피 지우는 값이라, 처음부터 "모르는 차원은 키를 생략"하도록 가르친다
  나누는 것
    - familyId(시드 프로필) 단위로 train/valid 를 나눈다. 같은 시드의 세션이 양쪽에 있으면 누수다

    .venv/bin/python scripts/build_training_data.py data/teacher/pilot200
    .venv/bin/python scripts/build_training_data.py data/teacher/pilot200 --out-dir data/train --valid-ratio 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "train"
MIN_AGREEMENT = 0.6


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def is_valid_family(family_id: str, ratio: float) -> bool:
    """familyId 해시로 고정 분할. 실행할 때마다, 데이터가 늘어도 같은 시드는 같은 쪽에 남는다."""
    bucket = int(hashlib.sha256(family_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < ratio


def _dumps(value: dict) -> str:
    # 운영 프롬프트의 예시 형식({"primary": [], ...})과 같은 구분자
    return json.dumps(value, ensure_ascii=False)


def tagging_example(row: dict, *, min_agreement: float) -> tuple[dict | None, str | None]:
    """teacher 태깅 한 줄 → (학습 예시, 버린 이유)."""
    from app.features.persona.agents import TAG_PROMPT

    agg = row.get("aggregate") or {}
    if not agg.get("validSamples"):
        return None, "failed"
    if agg.get("agreement", 0.0) < min_agreement:
        return None, "low_agreement"
    majority = agg["majority"]
    target = {
        "primary": [] if majority["off_topic"] else list(majority["primary"]),
        "secondary": [] if majority["off_topic"] else list(majority["secondary"]),
        "off_topic": bool(majority["off_topic"]),
    }
    user = f"질문: {row['input']['question']}\n답변: {row['input']['answer']}"  # 운영 TaggingAgent 와 같은 입력
    return {
        "messages": [
            {"role": "system", "content": TAG_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": _dumps(target)},
        ]
    }, None


def clean_extraction(raw: dict, answered: set[str]) -> dict:
    """허용 차원 밖 점수·텍스트와 스키마에 없는 키를 지운다. 순서는 teacher 원출력을 따른다."""
    from app.features.persona.schemas import SCORED, TEXTUAL, RawExtraction

    known = set(RawExtraction.model_fields)
    cleaned = {}
    for key, value in raw.items():
        if key not in known:
            continue
        if key in SCORED and (value is None or key not in answered):
            continue
        if key in TEXTUAL and (not value or key not in answered):
            continue
        cleaned[key] = value
    return cleaned


def extraction_example(row: dict) -> tuple[dict | None, str | None]:
    from app.features.persona.agents import RUBRIC, ExtractionAgent

    if "output" not in row:
        return None, "failed"
    if row.get("schemaError"):
        return None, "schema_error"
    answered = set(row["input"]["answered"])
    content = ExtractionAgent._transcript(row["input"]["history"]) + ExtractionAgent._answered_section(answered)
    return {
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": content},
            {"role": "assistant", "content": _dumps(clean_extraction(row["output"], answered))},
        ]
    }, None


def check_prompt_versions(rows: list[dict]) -> list[str]:
    """라벨을 만들 때의 프롬프트 해시가 지금 운영 프롬프트와 다른지."""
    from app.features.persona.agents import RUBRIC, TAG_PROMPT
    from scripts.label_with_teacher import prompt_version

    current = {"tagPromptVersion": prompt_version(TAG_PROMPT), "extractPromptVersion": prompt_version(RUBRIC)}
    drift = set()
    for row in rows:
        teacher = row.get("teacher") or {}
        for key, value in current.items():
            if key in teacher and teacher[key] != value:
                drift.add(f"{key}: 라벨 {teacher[key]} ≠ 현재 {value}")
    return sorted(drift)


def build(
    run_dir: Path,
    *,
    valid_ratio: float,
    min_agreement: float,
) -> tuple[dict[str, list[dict]], dict]:
    """→ ({"tagging_train": [...], ...}, 요약)."""
    outputs: dict[str, list[dict]] = {f"{t}_{s}": [] for t in ("tagging", "extraction") for s in ("train", "valid")}
    summary: dict = {}
    for task, make in (
        ("tagging", lambda r: tagging_example(r, min_agreement=min_agreement)),
        ("extraction", extraction_example),
    ):
        dropped: Counter = Counter()
        families: dict[str, set[str]] = {"train": set(), "valid": set()}
        for row in _rows(run_dir / f"{task}.jsonl"):
            if row.get("source") != "synthetic":
                dropped["gold_or_other_source"] += 1
                continue
            example, reason = make(row)
            if example is None:
                dropped[reason] += 1
                continue
            split = "valid" if is_valid_family(row["familyId"], valid_ratio) else "train"
            outputs[f"{task}_{split}"].append(example)
            families[split].add(row["familyId"])
        summary[task] = {
            "train": len(outputs[f"{task}_train"]),
            "valid": len(outputs[f"{task}_valid"]),
            "trainFamilies": len(families["train"]),
            "validFamilies": len(families["valid"]),
            "dropped": dict(sorted(dropped.items())),
        }
        if task == "tagging":
            summary[task]["offTopicTrain"] = sum(
                json.loads(e["messages"][-1]["content"])["off_topic"] for e in outputs["tagging_train"]
            )
    return outputs, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="label_with_teacher.py 출력 폴더 (tagging.jsonl, extraction.jsonl)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="검증 데이터로 보낼 familyId 비율")
    parser.add_argument("--min-agreement", type=float, default=MIN_AGREEMENT, help="태깅 일치율 하한")
    parser.add_argument("--allow-prompt-drift", action="store_true", help="프롬프트가 바뀌었어도 계속")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))

    drift = check_prompt_versions(_rows(args.run / "tagging.jsonl") + _rows(args.run / "extraction.jsonl"))
    if drift and not args.allow_prompt_drift:
        print("라벨을 만든 뒤 운영 프롬프트가 바뀌었습니다. 다시 라벨링하거나 --allow-prompt-drift:", file=sys.stderr)
        for line in drift:
            print(" -", line, file=sys.stderr)
        return 1

    outputs, summary = build(args.run, valid_ratio=args.valid_ratio, min_agreement=args.min_agreement)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in outputs.items():
        path = args.out_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8")
    summary |= {"source": str(args.run), "validRatio": args.valid_ratio, "minAgreement": args.min_agreement}
    summary["promptDrift"] = drift
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
