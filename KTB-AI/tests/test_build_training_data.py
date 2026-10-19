"""teacher 라벨 → 학습 데이터 변환 — 평가셋 누수와 운영 프롬프트 불일치를 막는 규칙을 고정한다."""

import asyncio
import json

from app.features.persona.agents import RUBRIC, TAG_PROMPT
from scripts import build_training_data as btd
from scripts import label_with_teacher as lwt


def _tag_row(row_id, family, *, primary=("anxiety",), off_topic=False, agreement=1.0, source="synthetic"):
    return {
        "kind": "tagging",
        "id": row_id,
        "source": source,
        "familyId": family,
        "input": {"question": "답이 늦으면 어때요?", "answer": "계속 확인해요."},
        "context": {},
        "aggregate": {
            "validSamples": 5,
            "majority": {"primary": list(primary), "secondary": [], "off_topic": off_topic},
            "agreement": agreement,
        },
        "teacher": {"tagPromptVersion": lwt.prompt_version(TAG_PROMPT)},
    }


def _extract_row(row_id, family, output, answered=("anxiety", "interests"), **extra):
    history = [{"role": "assistant", "content": "요즘 어때요?"}, {"role": "user", "content": "조약돌 모아요."}]
    return {
        "kind": "extraction",
        "id": row_id,
        "source": "synthetic",
        "familyId": family,
        "input": {"history": history, "answered": list(answered)},
        "output": output,
        "schemaError": None,
        "teacher": {"extractPromptVersion": lwt.prompt_version(RUBRIC)},
        **extra,
    }


def _write(run_dir, tags=(), extracts=()):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tagging.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tags))
    (run_dir / "extraction.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in extracts))


def test_tagging_example_matches_production_prompt_and_input():
    example, reason = btd.tagging_example(_tag_row("synthetic:s1:t0", "f1"), min_agreement=0.6)

    system, user, assistant = example["messages"]
    assert reason is None
    assert system == {"role": "system", "content": TAG_PROMPT}
    assert user["content"] == "질문: 답이 늦으면 어때요?\n답변: 계속 확인해요."
    assert json.loads(assistant["content"]) == {"primary": ["anxiety"], "secondary": [], "off_topic": False}


def test_off_topic_target_clears_dimensions():
    example, _ = btd.tagging_example(_tag_row("x", "f1", off_topic=True), min_agreement=0.6)

    assert json.loads(example["messages"][-1]["content"]) == {"primary": [], "secondary": [], "off_topic": True}


def test_gold_low_agreement_and_failed_rows_are_dropped(tmp_path):
    failed = _tag_row("synthetic:s3:t0", "f1")
    failed["aggregate"] = {"validSamples": 0}
    _write(
        tmp_path,
        tags=[
            _tag_row("gold:persona_onboarding_tagging-005", "f1", source="gold"),
            _tag_row("synthetic:s2:t0", "f1", agreement=0.4),
            failed,
            _tag_row("synthetic:s4:t0", "f1"),
        ],
    )

    outputs, summary = btd.build(tmp_path, valid_ratio=0.0, min_agreement=0.6)

    assert len(outputs["tagging_train"]) == 1
    assert summary["tagging"]["dropped"] == {"failed": 1, "gold_or_other_source": 1, "low_agreement": 1}


def test_extraction_target_drops_unanswered_and_unknown_keys():
    raw = {
        "anxiety": 82,
        "avoidance": 50,  # answered 밖 → 지움
        "interests": ["조약돌 분류"],
        "routine": ["저녁 정리"],  # answered 밖 → 지움
        "made_up": 1,  # 스키마에 없음 → 지움
        "narrative": {"headline": "h"},
    }

    example, _ = btd.extraction_example(_extract_row("synthetic:s1", "f1", raw))

    system, user, assistant = example["messages"]
    assert system["content"] == RUBRIC
    assert user["content"].startswith("하루: 요즘 어때요?\n사용자: 조약돌 모아요.\n\n# 사용자가 직접 답한 항목")
    assert json.loads(assistant["content"]) == {
        "anxiety": 82,
        "interests": ["조약돌 분류"],
        "narrative": {"headline": "h"},
    }


def test_schema_error_rows_are_dropped():
    example, reason = btd.extraction_example(_extract_row("s", "f", {"anxiety": 150}, schemaError="range"))

    assert example is None and reason == "schema_error"


def test_split_keeps_each_family_on_one_side(tmp_path):
    tags = [_tag_row(f"synthetic:s{i}:t{t}", f"family-{i}") for i in range(40) for t in range(3)]
    _write(tmp_path, tags=tags)

    outputs, summary = btd.build(tmp_path, valid_ratio=0.3, min_agreement=0.6)

    assert summary["tagging"]["trainFamilies"] + summary["tagging"]["validFamilies"] == 40
    assert summary["tagging"]["validFamilies"] > 0
    assert len(outputs["tagging_valid"]) == 3 * summary["tagging"]["validFamilies"]


def test_prompt_drift_is_detected():
    row = _tag_row("x", "f")
    row["teacher"]["tagPromptVersion"] = "000000000000"

    assert btd.check_prompt_versions([row])


def test_end_to_end_from_label_with_teacher_output(tmp_path):
    """label_with_teacher.run 이 쓰는 실제 형식을 그대로 읽는지 (가짜 teacher)."""

    async def fake(system, user, temperature, max_tokens):
        if system == TAG_PROMPT:
            return lwt.Completion('{"primary": ["anxiety"], "secondary": [], "off_topic": false}')
        return lwt.Completion('{"anxiety": 80, "avoidance": 40}')

    tags = [lwt.TagItem(f"synthetic:s{i}:t0", "synthetic", f"f{i}", "질문", "답변") for i in range(3)]
    extracts = [
        lwt.ExtractItem(f"synthetic:s{i}", "synthetic", f"f{i}", [{"role": "user", "content": "답변"}], None, [t.id])
        for i, t in enumerate(tags)
    ]
    asyncio.run(
        lwt.run(
            tags,
            extracts,
            fake,
            out_dir=tmp_path,
            samples=3,
            temperature=0.7,
            concurrency=2,
            resume=False,
            teacher_meta={"model": "fake"},
        )
    )

    outputs, summary = btd.build(tmp_path, valid_ratio=0.0, min_agreement=0.6)

    assert summary["tagging"]["train"] == 3 and summary["extraction"]["train"] == 3
    # answered = 태깅 다수결 primary → anxiety 만 남고 avoidance 는 지워진다
    assert json.loads(outputs["extraction_train"][0]["messages"][-1]["content"]) == {"anxiety": 80}
    assert not btd.check_prompt_versions([json.loads(x) for x in (tmp_path / "tagging.jsonl").read_text().splitlines()])
