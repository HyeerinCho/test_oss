"""하루 질문 다시 쓰기 — 시드 질문으로 받은 합성 세션의 질문을 운영 ConversationAgent 로 바꾼다.

subscription_simulator.py 는 주제의 시드 질문에 답을 받는다. 운영에서는 하루가 직전 답에 반응하고 주제로 넘어가므로,
턴마다 그때까지의 대화(다시 쓴 질문 + 받은 답)를 넣어 운영과 같은 방식으로 질문을 만든다.
API 생성기(generate_synthetic_onboarding.py)가 질문을 먼저 만들고 답을 받는 것과 넣는 정보는 같다.

  - 답·주제·답변 방식은 바꾸지 않는다. 원래 질문은 turns[].seedQuestion 에 남긴다
  - 되묻기(reask) 질문은 운영처럼 시드 질문 그대로다 (LLM 을 부르지 않는다)
  - 질문 생성이 실패하면 운영처럼 시드 질문으로 대신한다(questionSource=seed). 그 수를 요약에 남긴다
  - 세션마다 다 끝나면 바로 출력에 붙인다. 끊겨도 --resume 으로 남은 세션만 이어서 한다

    uv run python scripts/rewrite_questions.py data/synthetic/sub-pilot40-v2/sessions.jsonl
    uv run python scripts/rewrite_questions.py data/synthetic/sub-1000/sessions.jsonl --concurrency 8 --resume
    uv run python scripts/rewrite_questions.py data/synthetic/sub-pilot40-v2/sessions.jsonl --dry-run   # LLM 없이

모델은 운영과 같은 LLM_* 설정(.env)이다. Langfuse 트레이스는 운영 지표를 오염시키지 않도록 기본으로 끈다 (--trace 로 켠다).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_onboarding import (  # noqa: E402
    QuestionFn,
    Usage,
    estimate_cost,
    make_question_fn,
    seed_question,
)

REWRITE_VERSION = "question-rewrite/1"


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


async def rewrite_session(record: dict, ask: QuestionFn, total_turns: int) -> tuple[dict, int, int]:
    """세션 하나의 질문을 턴 순서대로 다시 쓴다. (새 레코드, LLM 질문 수, 시드 폴백 수)"""
    from app.features.persona.schemas import TOPICS_BY_ID

    out = json.loads(json.dumps(record))  # 원본을 건드리지 않는다
    nickname = out["seedProfile"]["nickname"]
    history: list[dict] = []
    llm = fallback = 0
    for t in sorted(out["turns"], key=lambda x: x["turnIndex"]):
        question, source = await ask(
            history=history,
            topic=TOPICS_BY_ID[t["topicId"]],
            turn_index=t["turnIndex"],
            total_turns=total_turns,
            nickname=nickname,
        )
        t.setdefault("seedQuestion", t["question"])
        t["question"], t["questionSource"] = question, source
        if source == "llm":
            llm += 1
        else:
            fallback += 1
        # PersonaService._history 와 같은 모양. 되묻기 전 무관 답은 운영처럼 넣지 않는다
        history += [{"role": "assistant", "content": question}, {"role": "user", "content": t["answer"]}]
    out["history"] = history
    usage = out.setdefault("usage", {})
    usage["conversation_calls"] = usage.get("conversation_calls", 0) + llm
    return out, llm, fallback


async def run(args: argparse.Namespace) -> dict:
    from app.core.config import get_settings
    from app.features.persona.service import ONBOARDING_TOTAL_TURNS

    total_turns = args.total_turns or ONBOARDING_TOTAL_TURNS
    if args.dry_run:
        ask: QuestionFn = seed_question
        model = "seed"
    else:
        from app.features.persona.agents import ConversationAgent

        ask = make_question_fn(ConversationAgent())
        model = get_settings().llm_model

    src, out = Path(args.input), Path(args.out) if args.out else Path(args.input).with_suffix(".rewritten.jsonl")
    records = _rows(src)
    done: set[str] = set()
    if out.exists():
        if not args.resume:
            raise SystemExit(f"{out} 가 이미 있습니다. 이어서 하려면 --resume, 새로 하려면 파일을 지우세요.")
        done = {r["sessionId"] for r in _rows(out)}
    todo = [r for r in records if r["sessionId"] not in done][: args.limit]
    out.parent.mkdir(parents=True, exist_ok=True)

    stamp = {
        "version": REWRITE_VERSION,
        "conversationModel": model,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    totals = {"written": 0, "llmQuestions": 0, "seedFallbacks": 0}
    failures: list[dict] = []

    async def one(record: dict) -> None:
        async with sem:
            try:
                new, llm, fallback = await rewrite_session(record, ask, total_turns)
            except Exception as e:  # 한 세션 실패로 전체를 멈추지 않는다 — --resume 으로 다시 한다
                failures.append({"sessionId": record["sessionId"], "error": f"{type(e).__name__}: {e}"})
                return
        new["generator"] = {**new.get("generator", {}), "conversationModel": model, "questionRewrite": stamp}
        async with lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(new, ensure_ascii=False) + "\n")
            totals["written"] += 1
            totals["llmQuestions"] += llm
            totals["seedFallbacks"] += fallback
        print(f"  [{totals['written'] + len(failures)}/{len(todo)}] {record['sessionId']} ok", file=sys.stderr)

    started = time.monotonic()
    await asyncio.gather(*(one(r) for r in todo))

    # 입력 순서로 다시 정렬해 둔다 (동시 실행이라 붙는 순서가 섞인다)
    order = {r["sessionId"]: i for i, r in enumerate(records)}
    rows = sorted(_rows(out), key=lambda r: order.get(r["sessionId"], len(order)))
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    summary = {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": str(src),
        "out": str(out),
        "sessions": len(records),
        "alreadyDone": len(done),
        **totals,
        "failed": len(failures),
        "remaining": len(records) - len(rows),
        "elapsedSeconds": round(time.monotonic() - started, 1),
        "conversationModel": model,
        "cost": estimate_cost(Usage(conversation_calls=totals["llmQuestions"]), args.price_in, args.price_out),
        "failures": failures,
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", help="subscription_simulator.py import 가 만든 sessions.jsonl")
    parser.add_argument("--out", default=None, help="기본 <input>.rewritten.jsonl (옆에 .summary.json)")
    parser.add_argument("--resume", action="store_true", help="출력에 이미 있는 세션은 건너뛰고 이어서 한다")
    parser.add_argument("--limit", type=int, default=None, help="이번에 할 세션 수 (시험용)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--total-turns", type=int, default=None, help="기본은 운영과 같은 ONBOARDING_TOTAL_TURNS")
    parser.add_argument("--dry-run", action="store_true", help="LLM 없이 시드 질문으로 흐름만 확인")
    parser.add_argument("--trace", action="store_true", help="Langfuse 트레이스 켜기 (기본 끔)")
    parser.add_argument("--price-in", type=float, default=None, help="입력 100만 토큰당 USD (비용 추정용)")
    parser.add_argument("--price-out", type=float, default=None, help="출력 100만 토큰당 USD (비용 추정용)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.trace:
        # 합성 세션이 운영 Langfuse 지표에 섞이지 않게 한다. app import 전에 정해야 한다
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    summary: dict[str, Any] = asyncio.run(run(args))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, ensure_ascii=False, indent=2))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
