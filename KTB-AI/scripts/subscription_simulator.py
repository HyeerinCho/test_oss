"""구독(Gemini 앱) 사용자 시뮬레이터 — API 대신 채팅창에 붙여넣어 합성 온보딩 답변을 만든다.

generate_synthetic_onboarding.py 와 같은 시드 프로필·답변 방식·주제 순서를 쓰되, 턴마다 LLM 을 부르지 않고
세션 하나를 통째로 프롬프트로 내보낸다. 사람이 Gem 에 붙여넣고 받은 JSON 을 다시 읽어
generate_synthetic_onboarding.py 와 같은 형식의 sessions.jsonl 로 만든다. 그래서 check_synthetic_labels.py·
label_with_teacher.py 를 그대로 쓸 수 있다.

  - 질문은 주제의 시드 질문이다(questionSource=seed). 운영 모델로 다시 쓰는 건 다음 단계다.
  - API 버전(v2)처럼 턴마다 그 질문이 겨누는 성향만 보여 준다. 한 메시지에 모든 턴이 보이므로
    성향이 섞일 수 있다 — 파일럿에서 check_synthetic_labels.py 의 구간 일치율로 확인한다.
  - 같은 --seed 면 API 버전과 같은 프로필·답변 방식·주제 순서가 나온다.

    # 1) 내보내기: Gem 지침 + 세션별 프롬프트
    uv run python scripts/subscription_simulator.py export --n 40 --seed 0 --out-dir data/synthetic/sub-pilot40
    # 2) prompts/*.txt 를 Gem 에 붙여넣고, 답을 responses/ 에 .txt 로 저장 (이름 자유, 여러 세션 한 파일도 됨)
    # 3) 읽어 들이기: 검증·누수 검사 → sessions.jsonl, 다시 할 세션은 sessions.retry.txt
    uv run python scripts/subscription_simulator.py import --dir data/synthetic/sub-pilot40
    # 4) 다시 할 세션만 다시 내보내기 → prompts/retry-*.txt
    uv run python scripts/subscription_simulator.py export --out-dir data/synthetic/sub-pilot40 \\
        --only data/synthetic/sub-pilot40/sessions.retry.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_onboarding import (  # noqa: E402
    DEFAULT_SIM_MODEL,
    GENERATOR_VERSION,
    MODE_INSTRUCTION,
    LeakChecker,
    Mode,
    SeedProfile,
    SessionResult,
    TurnRecord,
    clean_answer,
    describe_profile_v2,
    focus_section,
    pick_mode,
    sample_profile,
    to_record,
    write_outputs,
)

MANIFEST = "manifest.json"
PROMPT_VERSION = "subscription-v1"

GEM_INSTRUCTIONS = """\
당신은 소개팅 앱 온보딩에서 AI "하루"와 처음 대화하는 20~30대 사용자를 연기합니다.
매번 한 사람(또는 여러 사람)의 [나라는 사람], [말투], [대화 계획]이 주어집니다.
대화 계획의 각 턴에 그 사람으로서 메신저에 치듯 답하세요.

## 지켜야 할 것
- 각 답변은 200자 이내, 줄바꿈 없이 메시지 하나. 실제 사람처럼 문장이 매끄럽지 않아도 됩니다
- 성향을 설명하듯 말하지 않습니다. 유형명·점수·성향 이름을 쓰지 않고 구체적인 습관·장면·최근 일로 드러냅니다
- 각 턴의 [이 턴의 내 성향]과 어긋나는 말을 하지 않습니다. 거기 적힌 수준(어느 쪽인지)이 드러나게 답합니다
- 각 턴은 그 턴의 [이 턴의 내 성향]만 씁니다. 다른 턴에 적힌 성향은 그 턴의 답에서 꺼내지 않습니다
- "잘 모름"으로 적힌 성향은 적힌 방식대로만 대답하고, 그 성향을 짐작할 수 있는 행동·장면·예시를 말하지 않습니다
- "상황에 따라 다름"이면 한쪽으로 단정하지 말고 둘 다 있다는 걸 드러냅니다
- 예시 문장은 그대로 쓰지 말고, 비슷한 나만의 경험·장면으로 바꿔 말합니다
- 앞 턴에서 한 말을 되풀이하지 않습니다. 앞 턴 답과 자연스럽게 이어지게 씁니다
- 관심사·데이트 목록을 나열하지 않습니다. 그 질문에 맞는 것 한두 개만 꺼냅니다
- 하루가 AI라는 걸 알고 있지만 굳이 언급하지 않습니다
- 각 턴의 [답변 방식]을 반드시 따릅니다
- 여러 사람이 주어지면 사람마다 따로 연기합니다. 다른 사람의 설정·답을 섞지 않습니다

## 출력
사람(session_id)마다 JSON 객체 하나를 한 줄로 출력합니다. 설명·머리말 금지.
{"session_id": "<입력의 session_id>", "turns": [{"turn": 1, "answer": "..."}, ...]}
- 되묻기가 있는 턴은 {"turn": n, "off_topic_answer": "<첫 질문에 한 딴소리>", "answer": "<되묻기에 대한 답>"} 으로 씁니다
- 모든 턴을 빠짐없이, 턴 번호 순서대로 씁니다
"""

NO_TRAIT = "없음 — [나라는 사람]에서 이 질문에 맞는 것 한두 개만 꺼내 답하기"


# ══ 계획 ══════════════════════════════════════════════════


@dataclass
class PlannedTurn:
    turn_index: int
    topic: Any  # app.features.persona.schemas.Topic
    mode: Mode
    focus: list[str]


@dataclass
class SessionPlan:
    session_id: str
    index: int
    profile: SeedProfile
    turns: list[PlannedTurn]


def session_id(run_seed: int, index: int) -> str:
    # to_record 의 sessionId 와 같아야 import 에서 이어 붙일 수 있다
    return f"syn-{run_seed}-{index:05d}"


def plan_session(run_seed: int, index: int, total_turns: int) -> SessionPlan:
    """run_session 과 같은 순서로 rng 를 써서 같은 프로필·답변 방식·주제 순서를 만든다."""
    from app.features.persona.service import Coverage, next_topic

    rng = random.Random(f"{run_seed}:{index}")
    profile = sample_profile(rng, seed_id=f"seed-{run_seed}-{index:05d}")
    coverage = Coverage()
    used: list[str] = []
    turns: list[PlannedTurn] = []
    for turn_index in range(total_turns):
        topic = next_topic(coverage, turn_index, total_turns, used)
        if topic is None:
            break
        mode = pick_mode(rng, profile.style, closing=topic.is_closing)
        turns.append(PlannedTurn(turn_index, topic, mode, [*topic.covers, *topic.also_touches]))
        coverage.apply(list(topic.covers), list(topic.also_touches))
        used.append(topic.id)
    return SessionPlan(session_id(run_seed, index), index, profile, turns)


def reask_question(topic: Any) -> str:
    from app.features.persona.service import REASK_PREFIX

    return f"{REASK_PREFIX} {topic.seed}"


def trait_lines(profile: SeedProfile, focus: list[str]) -> list[str]:
    from app.features.persona.schemas import SCORED

    if not any(d in SCORED for d in focus):
        return [NO_TRAIT]
    # 머리말과 공통 안내(예시 바꿔 말하기·다른 성향 금지)는 Gem 지침에 있어 턴마다 되풀이하지 않는다
    lines = focus_section(profile, focus).splitlines()[1:]
    return [x for x in lines if not x.startswith(("예시 문장은", "위에 없는 다른 성향은"))]


def render_session(plan: SessionPlan) -> str:
    out = [f"session_id: {plan.session_id}", "", describe_profile_v2(plan.profile), "", "[대화 계획]"]
    for t in plan.turns:
        question = t.topic.seed
        if t.topic.choices:
            question += f" (선택지 버튼: {' / '.join(t.topic.choices)} — 골라도 되고 말로 답해도 됨)"
        out += ["", f"턴 {t.turn_index + 1}", f"- 하루의 말: {question}", f"- 답변 방식: {MODE_INSTRUCTION[t.mode]}"]
        if t.mode == "off_topic":
            out += [
                f"- 되묻기: {reask_question(t.topic)}",
                f"- 되묻기 답변 방식: {MODE_INSTRUCTION['normal']}",
                "- 이 턴의 내 성향 (되묻기에 답할 때만):",
            ]
        else:
            out.append("- 이 턴의 내 성향:")
        out += [f"  {line}" for line in trait_lines(plan.profile, t.focus)]
    return "\n".join(out)


# ══ 내보내기 ══════════════════════════════════════════════


def _read_ids(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def export(args: argparse.Namespace) -> int:
    from app.features.persona.service import ONBOARDING_TOTAL_TURNS

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / MANIFEST
    if args.only:
        if not manifest_path.exists():
            print(f"{manifest_path} 가 없습니다. 처음 내보낸 폴더를 --out-dir 로 주세요.", file=sys.stderr)
            return 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        wanted = set(_read_ids(Path(args.only)))
        indices = [i for i in range(manifest["n"]) if session_id(manifest["runSeed"], i) in wanted]
        prefix = "retry-"
    else:
        if args.n is None:
            print("--n 이 필요합니다.", file=sys.stderr)
            return 1
        if manifest_path.exists():
            print(
                f"{manifest_path} 가 이미 있습니다. 다른 --out-dir 을 쓰거나 --only 로 다시 내보내세요.",
                file=sys.stderr,
            )
            return 1
        manifest = {
            "runSeed": args.seed,
            "n": args.n,
            "totalTurns": args.total_turns or ONBOARDING_TOTAL_TURNS,
            "generatorVersion": GENERATOR_VERSION,
            "promptVersion": PROMPT_VERSION,
            "createdAt": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        indices = list(range(args.n))
        prefix = ""

    plans = [plan_session(manifest["runSeed"], i, manifest["totalTurns"]) for i in indices]
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "responses").mkdir(exist_ok=True)
    if not args.only:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "gem_instructions.txt").write_text(GEM_INSTRUCTIONS, encoding="utf-8")

    per = max(1, args.per_message)
    for b, start in enumerate(range(0, len(plans), per), 1):
        body = "\n\n────────\n\n".join(render_session(p) for p in plans[start : start + per])
        (prompts_dir / f"{prefix}{b:04d}.txt").write_text(body + "\n", encoding="utf-8")
    print(f"세션 {len(plans)}개 → {prompts_dir} ({-(-len(plans) // per)}개 파일)")
    return 0


# ══ 읽어 들이기 ════════════════════════════════════════════


def extract_objects(text: str) -> list[dict]:
    """응답 텍스트에서 session_id·turns 가 있는 JSON 객체를 모두 찾는다. 코드블록·머리말·배열이 섞여도 된다."""
    decoder = json.JSONDecoder()
    found: list[dict] = []
    i = 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and "session_id" in obj and "turns" in obj:
            found.append(obj)
            i = end
        else:
            i += 1
    return found


def validate(plan: SessionPlan, obj: dict) -> tuple[list[TurnRecord], list[str]]:
    """계획과 맞춰 TurnRecord 를 만든다. 문제가 있으면 이유 목록을 돌려준다 (그 세션은 다시 생성)."""
    from app.features.persona.schemas import MAX_ANSWER_LEN

    issues: list[str] = []
    by_turn: dict[int, dict] = {}
    for t in obj.get("turns") or []:
        if isinstance(t, dict) and isinstance(t.get("turn"), int):
            by_turn[t["turn"]] = t

    def text_of(raw: Any, where: str) -> str:
        if not isinstance(raw, str) or not raw.strip():
            issues.append(f"{where}: 빈 답")
            return ""
        if len(" ".join(raw.split())) > MAX_ANSWER_LEN:
            issues.append(f"{where}: {MAX_ANSWER_LEN}자 초과")
        return clean_answer(raw)

    records: list[TurnRecord] = []
    for p in plan.turns:
        n = p.turn_index + 1
        t = by_turn.get(n)
        if t is None:
            issues.append(f"턴 {n}: 없음")
            continue
        answer = text_of(t.get("answer"), f"턴 {n}")
        reask = None
        if p.mode == "off_topic":
            off = text_of(t.get("off_topic_answer"), f"턴 {n} 딴소리")
            reask = {"offTopicAnswer": off, "question": reask_question(p.topic)}
        records.append(TurnRecord(n - 1, p.topic.id, p.topic.seed, "seed", answer, p.mode, reask))
    extra = sorted(set(by_turn) - {p.turn_index + 1 for p in plan.turns})
    if extra:
        issues.append(f"계획에 없는 턴: {extra}")
    return records, issues


def import_responses(args: argparse.Namespace) -> int:
    base = Path(args.dir)
    manifest = json.loads((base / MANIFEST).read_text(encoding="utf-8"))
    run_seed, n, total_turns = manifest["runSeed"], manifest["n"], manifest["totalTurns"]
    ids = {session_id(run_seed, i): i for i in range(n)}
    models = {
        "simulatorModel": args.sim_model,
        "simulatorPrompt": manifest.get("promptVersion", PROMPT_VERSION),
        "simulatorChannel": "subscription",
        "conversationModel": "seed",
    }

    checker = LeakChecker()
    records: dict[int, dict] = {}
    problems: dict[str, list[str]] = {}
    unknown: list[str] = []
    files = sorted(p for p in (base / "responses").glob("*") if p.is_file())
    # 같은 세션이 여러 번 오면(다시 생성) 뒤 파일의 유효한 답이 앞을 덮는다. 유효한 답이 하나라도 있으면 그걸 쓴다
    for path in files:
        for obj in extract_objects(path.read_text(encoding="utf-8")):
            sid = str(obj["session_id"]).strip()
            if sid not in ids:
                unknown.append(sid)
                continue
            plan = plan_session(run_seed, ids[sid], total_turns)
            turns, issues = validate(plan, obj)
            result = SessionResult(profile=plan.profile, turns=turns)
            if not issues:
                leaks = checker.session_leaks(result)
                if leaks:
                    issues = [f"골드셋 누수: 턴 {x['turnIndex'] + 1}" for x in leaks]
            if issues:
                if plan.index not in records:
                    problems[sid] = [f"{path.name}: {x}" for x in issues]
                continue
            records[plan.index] = to_record(result, index=plan.index, run_seed=run_seed, models=models)
            problems.pop(sid, None)

    missing = [sid for sid, i in ids.items() if i not in records and sid not in problems]
    retry = sorted([*missing, *problems])
    out = Path(args.out) if args.out else base / "sessions.jsonl"
    summary = {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "out": str(out),
        "requested": n,
        "written": len(records),
        "missing": missing,
        "invalid": problems,
        "unknownSessionIds": sorted(set(unknown)),
        "responseFiles": [p.name for p in files],
        "models": models,
    }
    write_outputs(out, [records[i] for i in sorted(records)], summary)
    retry_path = out.with_suffix(".retry.txt")
    retry_path.write_text("".join(f"{x}\n" for x in retry), encoding="utf-8")

    print(f"유효 {len(records)}/{n} → {out}")
    print(f"응답 없음 {len(missing)} · 다시 생성 필요 {len(problems)} · 모르는 session_id {len(set(unknown))}")
    for sid, reasons in list(problems.items())[:10]:
        print(f"  {sid}: {'; '.join(reasons)}")
    if retry:
        print(f"다시 할 세션 목록 → {retry_path}")
        print(
            f"  uv run python scripts/subscription_simulator.py export --out-dir {base} --only {retry_path}",
        )
    return 0 if records else 1


# ══ 실행 ══════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("export", help="Gem 지침과 세션별 프롬프트를 파일로 만든다")
    ex.add_argument("--out-dir", required=True, help="이 실행의 폴더 (manifest·prompts·responses 가 생긴다)")
    ex.add_argument("--n", type=int, default=None, help="만들 세션 수")
    ex.add_argument("--seed", type=int, default=0, help="실행 시드. 같으면 API 버전과 같은 프로필·답변 방식")
    ex.add_argument("--total-turns", type=int, default=None, help="기본은 운영과 같은 ONBOARDING_TOTAL_TURNS")
    ex.add_argument("--per-message", type=int, default=1, help="프롬프트 파일 하나에 넣을 세션 수 (1~3 권장)")
    ex.add_argument("--only", default=None, help="이 파일의 session_id 만 다시 내보낸다 (import 의 .retry.txt)")

    im = sub.add_parser("import", help="responses/ 의 답을 검증해 sessions.jsonl 로 만든다")
    im.add_argument("--dir", required=True, help="export 의 --out-dir")
    im.add_argument("--out", default=None, help="기본 <dir>/sessions.jsonl (옆에 .summary.json·.retry.txt)")
    im.add_argument("--sim-model", default=DEFAULT_SIM_MODEL, help="답을 만든 모델 (기록용)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # 운영 Langfuse 지표에 섞이지 않게 한다. app import 전에 정해야 한다
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
    return export(args) if args.command == "export" else import_responses(args)


if __name__ == "__main__":
    raise SystemExit(main())
