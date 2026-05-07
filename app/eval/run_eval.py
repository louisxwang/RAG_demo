from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from json_repair import repair_json
from tqdm import tqdm

from app.agent.orchestrator import Orchestrator
from app.llm.client import LLMClient

log = logging.getLogger(__name__)


JUDGE_PROMPT = """You are an evaluator for a RAG system.
Given a QUESTION, a REFERENCE_ANSWER, the SYSTEM_ANSWER, and the RETRIEVED_CONTEXT, score:
- correctness (1-5): does the system answer match the reference answer?
- faithfulness (1-5): is the system answer supported by retrieved context (no hallucinations)?

Return ONLY valid JSON:
{
  "correctness": {"score": 1, "reason": "..."},
  "faithfulness": {"score": 1, "reason": "..."}
}
"""


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _exact_match(pred: str, ref: str) -> int:
    return int(_normalize(pred) == _normalize(ref))


def _f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p or not r:
        return 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in r:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def _llm_json(llm: LLMClient, messages: list[dict[str, Any]]) -> Any:
    txt = llm.chat(messages, temperature=0.0)
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        return json.loads(repair_json(txt))


def run_eval(eval_set_path: str, out_path: str = "eval/results.jsonl", use_llm_judge: bool = True) -> dict[str, float]:
    agent = Orchestrator()
    judge = LLMClient() if use_llm_judge else None

    src = Path(eval_set_path)
    if not src.exists():
        raise FileNotFoundError(eval_set_path)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    em_sum = 0
    f1_sum = 0.0
    judge_corr = 0.0
    judge_faith = 0.0
    judge_n = 0

    with src.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Evaluating"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = item["question"]
            ref = item["answer"]

            ans, ctx, steps = agent.run(q)

            em = _exact_match(ans, ref)
            f1 = _f1(ans, ref)

            record: dict[str, Any] = {
                "question": q,
                "reference_answer": ref,
                "system_answer": ans,
                "metrics": {"exact_match": em, "f1": f1},
                "retrieved_context": ctx,
                "agent_steps": steps,
            }

            if judge:
                j = _llm_json(
                    judge,
                    [
                        {"role": "system", "content": "You are a strict evaluator. Output ONLY valid JSON."},
                        {
                            "role": "user",
                            "content": (
                                JUDGE_PROMPT
                                + f"\nQUESTION:\n{q}\n\nREFERENCE_ANSWER:\n{ref}\n\nSYSTEM_ANSWER:\n{ans}\n\nRETRIEVED_CONTEXT:\n"
                                + "\n\n---\n\n".join(ctx[:6])
                            ),
                        },
                    ],
                )
                try:
                    c = float(j["correctness"]["score"])
                    fa = float(j["faithfulness"]["score"])
                    record["llm_judge"] = j
                    judge_corr += c
                    judge_faith += fa
                    judge_n += 1
                except Exception:  # noqa: BLE001
                    record["llm_judge"] = {"error": "parse_failed", "raw": j}

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            n += 1
            em_sum += em
            f1_sum += f1

    summary = {
        "n": float(n),
        "exact_match": (em_sum / n) if n else 0.0,
        "f1": (f1_sum / n) if n else 0.0,
    }
    if judge_n:
        summary["llm_judge_correctness_avg"] = judge_corr / judge_n
        summary["llm_judge_faithfulness_avg"] = judge_faith / judge_n

    log.info("Eval summary: %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    p = argparse.ArgumentParser(description="Run evaluation of your RAG pipeline on an eval set.")
    p.add_argument("--eval-set", required=True, help="Path to eval_set.jsonl")
    p.add_argument("--out", default="eval/results.jsonl")
    p.add_argument("--no-llm-judge", action="store_true", help="Disable LLM-as-judge scoring")
    args = p.parse_args()

    summary = run_eval(args.eval_set, args.out, use_llm_judge=not args.no_llm_judge)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

