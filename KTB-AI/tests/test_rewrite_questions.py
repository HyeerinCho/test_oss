"""질문 다시 쓰기 — LLM 없이 가짜 질문 함수로 턴 순서·대화 기록·이어 하기 규칙만 확인한다."""

import asyncio
import json
import random

import pytest

from app.features.persona.service import ONBOARDING_TOTAL_TURNS
from scripts.generate_synthetic_onboarding import FakeSimulator, run_session, sample_profile, seed_question, to_record
from scripts.rewrite_questions import main, rewrite_session


def _record(index: int, **style) -> dict:
    rng = random.Random(f"r:{index}")
    profile = sample_profile(rng, seed_id=f"seed-r-{index:05d}")
    for key, value in style.items():
        setattr(profile.style, key, value)
    result = asyncio.run(
        run_session(profile, rng=rng, ask=seed_question, simulator=FakeSimulator(), total_turns=ONBOARDING_TOTAL_TURNS)
    )
    return to_record(result, index=index, run_seed=0, models={"simulatorModel": "fake"})


def test_rewrite_feeds_rewritten_history_turn_by_turn():
    record = _record(0, off_topic_rate=1.0)  # 마지막 주제 말고는 모두 되묻기
    seen: list[list[dict]] = []

    async def ask(*, history, topic, turn_index, total_turns, nickname):
        seen.append(list(history))
        assert nickname == record["seedProfile"]["nickname"]
        return f"하루 질문 {turn_index}", "llm"

    new, llm, fallback = asyncio.run(rewrite_session(record, ask, ONBOARDING_TOTAL_TURNS))
    assert (llm, fallback) == (len(record["turns"]), 0)
    assert seen[0] == []
    assert seen[2][-2] == {"role": "assistant", "content": "하루 질문 1"}  # 앞 턴의 다시 쓴 질문이 들어간다
    for old, t in zip(record["turns"], new["turns"], strict=True):
        assert t["seedQuestion"] == old["question"]
        assert t["question"] == f"하루 질문 {t['turnIndex']}"
        assert t["answer"] == old["answer"]
        assert t["reask"] == old["reask"]  # 되묻기는 운영처럼 시드 질문 그대로
    assert new["history"][0]["content"] == "하루 질문 0"
    assert new["usage"]["conversation_calls"] == record["usage"]["conversation_calls"] + llm
    assert record["turns"][0]["question"] != "하루 질문 0"  # 원본은 그대로


def test_dry_run_writes_sorted_output_and_resume_skips_done(tmp_path, capsys):
    src = tmp_path / "sessions.jsonl"
    records = [_record(i) for i in range(3)]
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    assert main([str(src), "--dry-run", "--limit", "2"]) == 0
    out = tmp_path / "sessions.rewritten.jsonl"
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [r["sessionId"] for r in rows] == [r["sessionId"] for r in records[:2]]
    assert rows[0]["generator"]["questionRewrite"]["version"] == "question-rewrite/1"
    assert all(t["questionSource"] == "seed" for t in rows[0]["turns"])

    with pytest.raises(SystemExit):  # 출력이 이미 있으면 --resume 없이는 덮어쓰지 않는다
        main([str(src), "--dry-run"])

    assert main([str(src), "--dry-run", "--resume"]) == 0
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [r["sessionId"] for r in rows] == [r["sessionId"] for r in records]
    summary = json.loads((tmp_path / "sessions.rewritten.summary.json").read_text(encoding="utf-8"))
    assert (summary["alreadyDone"], summary["written"], summary["remaining"]) == (2, 1, 0)
    capsys.readouterr()
