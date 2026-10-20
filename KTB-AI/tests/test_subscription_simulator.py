"""구독 시뮬레이터 내보내기·읽어 들이기 — LLM 없이 계획·렌더링·검증 규칙만 확인한다."""

import asyncio
import json
import random

from app.features.persona.schemas import MAX_ANSWER_LEN
from scripts.generate_synthetic_onboarding import FakeSimulator, run_session, sample_profile, seed_question
from scripts.subscription_simulator import (
    NO_TRAIT,
    extract_objects,
    main,
    plan_session,
    render_session,
    session_id,
    validate,
)


def _fake_response(plan, *, skip_turn: int | None = None, long_turn: int | None = None) -> dict:
    turns = []
    for t in plan.turns:
        n = t.turn_index + 1
        if n == skip_turn:
            continue
        answer = "가" * (MAX_ANSWER_LEN + 1) if n == long_turn else f"{plan.session_id} 턴{n} 답이에요 ㅎㅎ"
        turn = {"turn": n, "answer": answer}
        if t.mode == "off_topic":
            turn["off_topic_answer"] = f"{plan.session_id} 턴{n} 아 배고프다"
        turns.append(turn)
    return {"session_id": plan.session_id, "turns": turns}


def _off_topic_index(run_seed: int) -> int:
    return next(i for i in range(200) if any(t.mode == "off_topic" for t in plan_session(run_seed, i, 10).turns))


def test_plan_matches_api_generator_for_same_seed():
    for index in range(5):
        plan = plan_session(3, index, 10)
        rng = random.Random(f"3:{index}")
        profile = sample_profile(rng, seed_id=f"seed-3-{index:05d}")
        result = asyncio.run(
            run_session(profile, rng=rng, ask=seed_question, simulator=FakeSimulator(), total_turns=10)
        )
        if "ㅋ" in profile.style.laughter:
            profile.style.laughter = ""
        assert plan.profile == profile
        assert [(t.topic.id, t.mode) for t in plan.turns] == [(t.topicId, t.mode) for t in result.turns]


def test_render_shows_only_turn_traits_and_reask():
    index = _off_topic_index(0)
    plan = plan_session(0, index, 10)
    text = render_session(plan)
    assert text.startswith(f"session_id: {plan.session_id}")
    assert text.count("\n턴 ") == len(plan.turns)
    assert NO_TRAIT in text  # 관심사 주제는 성향이 없다 — "모두 잘 모름"으로 짧게 답하라는 안내가 붙으면 안 된다
    assert "되묻기:" in text
    assert "예시 문장은 그대로" not in text  # 공통 안내는 Gem 지침에만
    for key in plan.profile.levels:
        assert key not in text  # 영문 차원 키가 새면 시뮬레이터가 그대로 말할 수 있다


def test_extract_objects_tolerates_fences_arrays_and_prose():
    text = (
        "여기 있어요!\n```json\n"
        '[{"session_id": "syn-0-00000", "turns": [{"turn": 1, "answer": "a"}]},'
        ' {"session_id": "syn-0-00001", "turns": []}]\n```\n'
        '{"session_id": "syn-0-00002", "turns": [{"turn": 1, "answer": "중괄호 { 있는 답"}]}'
    )
    assert [o["session_id"] for o in extract_objects(text)] == ["syn-0-00000", "syn-0-00001", "syn-0-00002"]


def test_export_import_round_trip_and_retry(tmp_path, capsys):
    out_dir = tmp_path / "run"
    n = _off_topic_index(0) + 1
    assert main(["export", "--n", str(n), "--seed", "0", "--out-dir", str(out_dir), "--per-message", "2"]) == 0
    assert (out_dir / "gem_instructions.txt").exists()
    assert len(list((out_dir / "prompts").glob("*.txt"))) == -(-n // 2)

    plans = [plan_session(0, i, 10) for i in range(n)]
    good = [_fake_response(p) for p in plans[2:]]
    bad = [_fake_response(plans[0], skip_turn=3), _fake_response(plans[1], long_turn=2)]
    (out_dir / "responses" / "a.txt").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in good + bad), encoding="utf-8"
    )
    assert main(["import", "--dir", str(out_dir)]) == 0

    rows = [json.loads(x) for x in (out_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["sessionId"] for r in rows] == [session_id(0, i) for i in range(2, n)]
    last = rows[-1]
    assert last["familyId"] == plans[-1].profile.seed_id
    assert last["generator"]["simulatorChannel"] == "subscription"
    assert len(last["history"]) == 2 * len(plans[-1].turns)
    off = [t for t in last["turns"] if t["mode"] == "off_topic"]
    assert off and all(t["reask"]["offTopicAnswer"] for t in off)

    retry = (out_dir / "sessions.retry.txt").read_text(encoding="utf-8").split()
    assert retry == [session_id(0, 0), session_id(0, 1)]
    summary = json.loads((out_dir / "sessions.summary.json").read_text(encoding="utf-8"))
    assert "턴 3: 없음" in summary["invalid"][session_id(0, 0)][0]
    assert f"{MAX_ANSWER_LEN}자 초과" in summary["invalid"][session_id(0, 1)][0]

    # 다시 할 세션만 다시 내보내고, 고친 답을 뒤 파일로 넣으면 덮어쓴다
    assert main(["export", "--out-dir", str(out_dir), "--only", str(out_dir / "sessions.retry.txt")]) == 0
    assert sorted(p.name for p in (out_dir / "prompts").glob("retry-*.txt")) == ["retry-0001.txt", "retry-0002.txt"]
    fixed = "\n".join(json.dumps(_fake_response(p), ensure_ascii=False) for p in plans[:2])
    (out_dir / "responses" / "b.txt").write_text(fixed, encoding="utf-8")
    assert main(["import", "--dir", str(out_dir)]) == 0
    rows = (out_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == n
    assert (out_dir / "sessions.retry.txt").read_text(encoding="utf-8") == ""
    capsys.readouterr()


def test_export_refuses_to_overwrite_run(tmp_path):
    out_dir = tmp_path / "run"
    assert main(["export", "--n", "1", "--out-dir", str(out_dir)]) == 0
    assert main(["export", "--n", "1", "--out-dir", str(out_dir)]) == 1


def test_k_laughter_is_removed_from_prompt_and_rejected_in_answers():
    index = next(i for i in range(200) if "ㅋ" in sample_profile(random.Random(f"0:{i}"), seed_id="s").style.laughter)
    plan = plan_session(0, index, 10)
    assert plan.profile.style.laughter == ""
    assert "ㅋ" not in render_session(plan)

    response = _fake_response(plan)
    response["turns"][0]["answer"] = "테니스 치는 게 제일 재밌어 ㅋㅋㅋ"
    _, issues = validate(plan, response)
    assert issues == ["턴 1: ㅋ 웃음 사용"]
