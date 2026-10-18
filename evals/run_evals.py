"""Phase 10: answer-quality evals over evals/golden_set.json.

Needs Qdrant with the indexed collection, and calls Groq: the generator for every case, the judge
for LLM metrics. Like the CLI it calls RAGService directly, so guardrails are not involved.
Run from the project root:
    python -m evals.run_evals --limit 3 --no-llm-metrics      # retrieval hits and refusals only
    python -m evals.run_evals --limit 3 --metrics faithfulness,correctness
    python -m evals.run_evals                                 # every case, every metric
    python -m evals.run_evals --resume evals/results/<file>.json    # continue a stopped run
    python -m evals.run_evals --threshold correctness=0.7     # override a threshold
    medrag eval ...                                           # same flags

Answer cases are scored on
  rec_hit     every expected recommendation is among the retrieved chunks (the same rule as
              scripts.retrieval_check: nice.find_rec); without `recs` (table answers), any
              retrieved chunk from expected_source
  title_hit   expected_source is among the cited titles (reported, not gated)
  metrics     DeepEval with the Groq judge: faithfulness, answer_relevancy, contextual_precision,
              contextual_recall and correctness (GEval: the answer against expected_answer)
Hard negatives (kind "refusal") make no judge calls. A correct refusal is confidence
insufficient_evidence, or an answer that only points to another guideline (a pointer phrase and
no doses).

The results file (evals/results/<timestamp>.json) is rewritten after every case. A judge or
generator rate limit that outlasts the retries (the daily token limit) stops the run with exit
code 2; --resume <file> then skips finished cases and only computes missing metrics, on the
stored answer. A Markdown summary is written next to the JSON. Exit codes: 0 every threshold
met, 1 below a threshold or a case failed, 2 stopped early on a rate limit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field, model_validator

from evals import JUDGE_DELAY_SECONDS
from src.config import get_settings
from src.core.llm import describe_error
from src.core.schemas import RAGResponse
from src.plugins.medrag.nice import find_rec

logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_SET = EVALS_DIR / "golden_set.json"
RESULTS_DIR = EVALS_DIR / "results"

METRIC_NAMES = ("faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall",
                "correctness")
SHORT_NAMES = {"faithfulness": "faith", "answer_relevancy": "relev", "contextual_precision": "c.prec",
               "contextual_recall": "c.rec", "correctness": "correct"}
# The run fails (exit 1) if a value is below its threshold. Override with --threshold name=value.
THRESHOLDS = {
    "rec_hit_rate": 0.8,
    "refusal_rate": 0.75,
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "contextual_precision": 0.6,
    "contextual_recall": 0.7,
    "correctness": 0.6,
}

# GEval steps (explicit, so DeepEval doesn't spend a judge call generating them from criteria).
CORRECTNESS_STEPS = [
    ("The expected output is the reference answer taken from the clinical guideline. Compare the "
     "actual output with it."),
    ("Heavily penalise every drug, treatment option, threshold, target, interval, duration or "
     "exception stated in the expected output that is missing from the actual output."),
    ("Heavily penalise anything in the actual output that contradicts the expected output, such as "
     "a different drug, threshold or recommendation."),
    ("Do not penalise correct extra detail that is consistent with the expected output, citation "
     "markers such as [1], or differences in wording."),
]

NO_CONTEXT_REASON = "nothing retrieved (question gate): scored 0 without a judge call"

_GUIDELINE_CODE = re.compile(r"^[A-Z]{1,3}\d+$")  # NG136 in "NICE NG136 Hypertension"
# "see NICE's guideline on hypertension in pregnancy", "covered by", "outside the scope".
_POINTER = re.compile(r"\bguidelines?\s+on\b|\b(?:covered|addressed)\s+(?:in|by)\b|"
                      r"\bnot\s+covered\b|\boutside\s+(?:the\s+)?scope\b", re.IGNORECASE)
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|g|micrograms?|mcg|units?|ml|mmol|iu)\b",
                   re.IGNORECASE)


class EvalError(Exception):
    """Bad input (golden set, resume file); reported without a traceback."""


class RateLimitStop(Exception):
    """A Groq 429 that outlasted the retries: save and stop, to be resumed later."""


# ---------- Golden set ----------

class GoldenCase(BaseModel):
    id: str
    kind: Literal["answer", "refusal"] = "answer"
    question: str
    expected_answer: str
    expected_source: str | None = None  # title, for answer cases
    recs: list[str] = []  # recommendation numbers; empty for table answers
    style: Literal["guideline", "clinician", "two_recs", "negative"]

    @model_validator(mode="after")
    def _answer_fields(self) -> GoldenCase:
        if self.kind == "answer" and not self.expected_source:
            raise ValueError(f"{self.id}: answer cases need expected_source")
        if self.recs and self.guideline is None:
            raise ValueError(f"{self.id}: no guideline code in expected_source")
        return self

    @property
    def guideline(self) -> str | None:
        return next((w for w in (self.expected_source or "").split()
                     if _GUIDELINE_CODE.match(w)), None)


def load_golden_set(path: Path = GOLDEN_SET) -> list[GoldenCase]:
    cases = [GoldenCase.model_validate(c) for c in json.loads(path.read_text(encoding="utf-8"))]
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise EvalError(f"duplicate golden-set ids: {', '.join(duplicates)}")
    return cases


# ---------- Scoring without an LLM ----------

def _same_title(a: str | None, b: str | None) -> bool:
    return bool(a and b) and a.casefold() == b.casefold()


def rec_hit(case: GoldenCase, nodes: Sequence[NodeWithScore]) -> bool:
    """Every expected recommendation is in a retrieved chunk; without recs, the expected source."""
    if case.recs:
        return all(find_rec(nodes, case.guideline, rec) is not None for rec in case.recs)
    return any(_same_title(case.expected_source, n.node.metadata.get("title")) for n in nodes)


def title_hit(case: GoldenCase, response: RAGResponse) -> bool:
    return any(_same_title(case.expected_source, c.title) for c in response.citations)


def is_refusal(response: RAGResponse) -> bool:
    """insufficient_evidence, or an answer that only points to another guideline."""
    if response.confidence == "insufficient_evidence":
        return True
    return bool(_POINTER.search(response.answer)) and not _DOSE.search(response.answer)


# ---------- Results ----------

class MetricResult(BaseModel):
    score: float | None = None
    success: bool | None = None
    reason: str | None = None
    error: str | None = None


class TokenUsage(BaseModel):
    prompt: int = 0
    completion: int = 0
    calls: int = 0
    estimated_calls: int = 0  # replies without `usage`: ~4 characters per token

    @property
    def total(self) -> int:
        return self.prompt + self.completion


class CaseResult(BaseModel):
    id: str
    kind: Literal["answer", "refusal"]
    style: str
    question: str
    answer: str | None = None
    confidence: str | None = None
    cited_titles: list[str] = []
    contexts: list[str] = []  # kept so --resume can score missing metrics without asking again
    latency_ms: int | None = None
    rec_hit: bool | None = None
    title_hit: bool | None = None
    refused: bool | None = None
    metrics: dict[str, MetricResult] = {}
    judge_tokens: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None  # RAGService.ask failed

    def missing_metrics(self, names: Iterable[str]) -> list[str]:
        if self.kind != "answer":
            return []
        return [n for n in names if n not in self.metrics or self.metrics[n].score is None]


class MetricSummary(BaseModel):
    mean: float | None
    n: int
    errors: int


class Check(BaseModel):
    value: float | None
    threshold: float
    passed: bool | None  # None: nothing to measure (e.g. no negatives within --limit)


class Summary(BaseModel):
    cases: int
    answer_cases: int
    negatives: int
    failed_cases: int
    rec_hit_rate: float | None
    title_hit_rate: float | None
    refusal_rate: float | None
    metrics: dict[str, MetricSummary]
    by_style: dict[str, dict[str, float | int | None]]
    judge_tokens: dict[str, float | int | None]
    checks: dict[str, Check]
    passed: bool


class RunFile(BaseModel):
    project: str
    generator_model: str
    judge_model: str
    metrics: list[str] = []
    thresholds: dict[str, float] = {}
    started: str
    updated: str | None = None
    cases: dict[str, CaseResult] = {}
    summary: Summary | None = None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save_run(state: RunFile, path: Path) -> None:
    """Write the whole file to a temp file, then swap it in: the file is always valid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated = _now()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(state.model_dump_json(indent=2))
    os.replace(tmp, path)


def load_run(path: Path) -> RunFile:
    try:
        return RunFile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalError(f"can't resume from {path}: {exc}") from exc


# ---------- Running a case ----------

def _is_rate_limit(exc: BaseException | None) -> bool:
    for _ in range(10):  # the exception chain, bounded
        if exc is None:
            return False
        if getattr(exc, "status_code", None) == 429:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _error_text(exc: BaseException) -> str:
    # API errors: type/status/code only (their messages can echo request details).
    if getattr(exc, "status_code", None) is not None:
        return describe_error(exc)
    return f"{type(exc).__name__}: {exc}"[:300]


def ask_case(case: GoldenCase, service) -> CaseResult:
    """Ask the question; score retrieval (answer cases) or refusal (hard negatives)."""
    result = CaseResult(id=case.id, kind=case.kind, style=case.style, question=case.question)
    try:
        response, nodes = service.ask_with_nodes(case.question)
    except Exception as exc:
        if _is_rate_limit(exc):
            raise RateLimitStop(describe_error(exc)) from exc
        logger.warning("%s: ask failed (%s)", case.id, _error_text(exc))
        result.error = _error_text(exc)
        return result
    result.answer = response.answer
    result.confidence = response.confidence
    result.cited_titles = [c.title for c in response.citations]
    result.contexts = list(response.contexts)
    result.latency_ms = response.latency_ms
    if case.kind == "answer":
        result.rec_hit = rec_hit(case, nodes)
        result.title_hit = title_hit(case, response)
    else:
        result.refused = is_refusal(response)
    return result


def _usage_of(judge) -> tuple[int, int, int, int]:
    usage = getattr(judge, "usage", None)
    if usage is None:
        return 0, 0, 0, 0
    return usage.prompt, usage.completion, usage.calls, usage.estimated_calls


def score_metrics(case: GoldenCase, result: CaseResult, metrics: dict, judge) -> None:
    """Fill result.metrics for `metrics` (name -> DeepEval metric), one after another."""
    if not result.contexts:
        for name in metrics:
            result.metrics[name] = MetricResult(score=0.0, success=False, reason=NO_CONTEXT_REASON)
        return
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(input=case.question, actual_output=result.answer or "",
                            expected_output=case.expected_answer,
                            retrieval_context=result.contexts)
    for name, metric in metrics.items():
        before = _usage_of(judge)
        try:
            metric.measure(test_case, _show_indicator=False)
            result.metrics[name] = MetricResult(score=metric.score,
                                                success=metric.is_successful(),
                                                reason=getattr(metric, "reason", None))
        except Exception as exc:
            if _is_rate_limit(exc):
                raise RateLimitStop(describe_error(exc)) from exc
            logger.warning("%s: %s failed (%s)", case.id, name, _error_text(exc))
            result.metrics[name] = MetricResult(error=_error_text(exc))
        finally:
            prompt, completion, calls, estimated = (
                a - b for a, b in zip(_usage_of(judge), before))
            tokens = result.judge_tokens
            tokens.prompt += prompt
            tokens.completion += completion
            tokens.calls += calls
            tokens.estimated_calls += estimated


def make_metrics(names: Sequence[str], judge, thresholds: dict[str, float]) -> dict:
    """DeepEval metrics graded by `judge`, sequential (async_mode=False). The built-in metrics
    skip their reason call (include_reason=False) to save judge tokens; GEval gives a reason in
    the same call as its score."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import SingleTurnParams

    common = {"model": judge, "async_mode": False}
    builders: dict[str, Callable[[float], object]] = {
        "faithfulness": lambda t: FaithfulnessMetric(threshold=t, include_reason=False, **common),
        "answer_relevancy": lambda t: AnswerRelevancyMetric(threshold=t, include_reason=False,
                                                            **common),
        "contextual_precision": lambda t: ContextualPrecisionMetric(threshold=t,
                                                                    include_reason=False, **common),
        "contextual_recall": lambda t: ContextualRecallMetric(threshold=t, include_reason=False,
                                                              **common),
        "correctness": lambda t: GEval(
            name="Correctness",
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                               SingleTurnParams.EXPECTED_OUTPUT],
            evaluation_steps=CORRECTNESS_STEPS, threshold=t, **common),
    }
    return {name: builders[name](thresholds[name]) for name in names}


# ---------- Summary ----------

def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def summarize(results: Sequence[CaseResult], metric_names: Sequence[str],
              thresholds: dict[str, float], golden_answer_cases: int) -> Summary:
    answers = [r for r in results if r.kind == "answer"]
    negatives = [r for r in results if r.kind == "refusal"]
    rate = lambda flags: _mean(None if f is None else float(f) for f in flags)

    metrics = {}
    for name in metric_names:
        scored = [r.metrics[name] for r in answers if name in r.metrics]
        metrics[name] = MetricSummary(mean=_mean(m.score for m in scored),
                                      n=sum(m.score is not None for m in scored),
                                      errors=sum(m.error is not None for m in scored))
    by_style = {}
    for style in sorted({r.style for r in answers}):
        group = [r for r in answers if r.style == style]
        by_style[style] = {"cases": len(group), "rec_hit_rate": rate(r.rec_hit for r in group),
                           **{name: _mean(r.metrics[name].score for r in group
                                          if name in r.metrics) for name in metric_names}}

    values = {"rec_hit_rate": rate(r.rec_hit for r in answers),
              "refusal_rate": rate(r.refused for r in negatives),
              **{name: metrics[name].mean for name in metric_names}}
    checks = {}
    for name, value in values.items():
        if value is None:  # only errors -> fail; no data at all -> not checked
            passed = False if name in metrics and metrics[name].errors else None
        else:
            passed = value >= thresholds[name]
        checks[name] = Check(value=value, threshold=thresholds[name], passed=passed)

    judged = [r for r in answers if r.judge_tokens.calls]
    per_case = _mean(r.judge_tokens.total for r in judged)
    judge_tokens = {
        "total": sum(r.judge_tokens.total for r in judged),
        "cases": len(judged),
        "mean_per_case": round(per_case) if per_case is not None else None,
        "estimated_calls": sum(r.judge_tokens.estimated_calls for r in judged),
        "calls": sum(r.judge_tokens.calls for r in judged),
        "projected_full_set": round(per_case * golden_answer_cases) if per_case else None,
    }
    failed = sum(r.error is not None for r in results)
    return Summary(cases=len(results), answer_cases=len(answers), negatives=len(negatives),
                   failed_cases=failed, rec_hit_rate=values["rec_hit_rate"],
                   title_hit_rate=rate(r.title_hit for r in answers),
                   refusal_rate=values["refusal_rate"], metrics=metrics, by_style=by_style,
                   judge_tokens=judge_tokens, checks=checks,
                   passed=failed == 0 and all(c.passed is not False for c in checks.values()))


def _fmt(value: float | bool | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _pass(check: Check) -> str:
    return {True: "pass", False: "FAIL", None: "n/a"}[check.passed]


def render_markdown(state: RunFile, summary: Summary, title: str) -> str:
    names = state.metrics
    lines = [
        f"# Eval run {title}",
        "",
        (f"- Project: `{state.project}`; generator `{state.generator_model}`, judge "
         f"`{state.judge_model}`"),
        f"- Metrics: {', '.join(names) or 'none (--no-llm-metrics)'}",
        (f"- Cases: {summary.cases} ({summary.answer_cases} answer, {summary.negatives} hard "
         f"negatives, {summary.failed_cases} failed); started {state.started}, updated "
         f"{state.updated}"),
        f"- Result: **{'PASS' if summary.passed else 'FAIL'}**",
        "",
        "## Checks",
        "",
        "| Check | Value | Threshold | Result |",
        "|---|---|---|---|",
    ]
    lines += [f"| {name} | {_fmt(c.value)} | {c.threshold:.2f} | {_pass(c)} |"
              for name, c in summary.checks.items()]
    lines += [f"| title_hit_rate (not gated) | {_fmt(summary.title_hit_rate)} | | |", ""]

    lines += ["## By style", "",
              "| Style | Cases | rec hit | " + " | ".join(SHORT_NAMES[n] for n in names) + " |",
              "|---|---|---|" + "---|" * len(names)]
    for style, row in summary.by_style.items():
        lines.append(f"| {style} | {row['cases']} | {_fmt(row['rec_hit_rate'])} | "
                     + " | ".join(_fmt(row[n]) for n in names) + " |")

    tokens = summary.judge_tokens
    lines += ["", "## Judge tokens", "",
              (f"{_fmt(tokens['total'])} tokens over {tokens['cases']} cases "
               f"({tokens['calls']} calls, {tokens['estimated_calls']} estimated): "
               f"~{_fmt(tokens['mean_per_case'])} per case, ~{_fmt(tokens['projected_full_set'])} "
               "projected for every answer case in the golden set with these metrics."), ""]

    lines += ["## Cases", "",
              "| Case | Style | rec hit | title hit | refused | confidence | "
              + " | ".join(SHORT_NAMES[n] for n in names) + " | judge tokens | error |",
              "|---|---|---|---|---|---|" + "---|" * len(names) + "---|---|"]
    for r in state.cases.values():
        scores = [_fmt(r.metrics[n].score) if n in r.metrics and r.metrics[n].error is None
                  else ("err" if n in r.metrics else "-") for n in names]
        lines.append(f"| {r.id} | {r.style} | {_fmt(r.rec_hit)} | {_fmt(r.title_hit)} | "
                     f"{_fmt(r.refused)} | {r.confidence or '-'} | " + " | ".join(scores)
                     + f" | {r.judge_tokens.total or '-'} | {r.error or ''} |")
    return "\n".join(lines) + "\n"


def format_summary(summary: Summary) -> str:
    lines = [(f"\n{summary.cases} cases ({summary.answer_cases} answer, {summary.negatives} "
             f"negatives, {summary.failed_cases} failed)")]
    lines += [f"  {name:<22} {_fmt(c.value):>5}  (threshold {c.threshold:.2f})  {_pass(c)}"
              for name, c in summary.checks.items()]
    lines.append(f"  {'title_hit_rate':<22} {_fmt(summary.title_hit_rate):>5}  (not gated)")
    tokens = summary.judge_tokens
    if tokens["cases"]:
        lines.append(f"Judge tokens: {tokens['total']} over {tokens['cases']} cases, "
                     f"~{tokens['mean_per_case']} per case ({tokens['estimated_calls']} of "
                     f"{tokens['calls']} calls estimated); ~{tokens['projected_full_set']} for "
                     "every answer case with these metrics")
    lines.append("PASS" if summary.passed else "FAIL")
    return "\n".join(lines)


def _progress(i: int, n: int, r: CaseResult) -> str:
    if r.error:
        return f"[{i}/{n}] {r.id}: ask failed ({r.error})"
    if r.kind == "refusal":
        return f"[{i}/{n}] {r.id}: refused={_fmt(r.refused)} ({r.confidence})"
    scores = " ".join(f"{SHORT_NAMES[k]}={_fmt(m.score) if m.error is None else 'err'}"
                      for k, m in r.metrics.items())
    tokens = f" ({r.judge_tokens.total} judge tokens)" if r.judge_tokens.calls else ""
    return (f"[{i}/{n}] {r.id}: rec_hit={_fmt(r.rec_hit)} title_hit={_fmt(r.title_hit)} "
            f"{scores}{tokens}").rstrip()


# ---------- Run ----------

def run(project: str = "medrag", *, limit: int | None = None,
        metrics: Sequence[str] = METRIC_NAMES, resume: Path | None = None,
        judge_delay: float = JUDGE_DELAY_SECONDS, thresholds: dict[str, float] | None = None,
        golden: Path = GOLDEN_SET, out_dir: Path = RESULTS_DIR, service=None, judge=None,
        metric_factory: Callable[..., dict] = make_metrics) -> int:
    """Evaluate the golden set; returns the exit code (0 pass, 1 fail, 2 stopped early)."""
    settings = get_settings()
    metrics = [m for m in METRIC_NAMES if m in set(metrics)]
    thresholds = {**THRESHOLDS, **(thresholds or {})}
    golden_cases = load_golden_set(golden)
    cases = golden_cases[:limit] if limit else golden_cases

    if metrics and judge is None:
        from evals.judge import GroqJudge
        judge = GroqJudge(delay=judge_delay)
    judge_model = judge.get_model_name() if judge is not None else settings.groq_judge_model

    if resume is not None:
        path = Path(resume)
        state = load_run(path)
        if state.project != project:
            raise EvalError(f"{path} is a run of project '{state.project}', not '{project}'")
        judged = any(r.metrics for r in state.cases.values())
        if metrics and judged and state.judge_model != judge_model:
            raise EvalError(f"{path} was judged by {state.judge_model}, not {judge_model}; "
                            "mixing judges makes the scores incomparable")
        if metrics:
            state.judge_model = judge_model
        state.metrics = [m for m in METRIC_NAMES if m in set(state.metrics) | set(metrics)]
    else:
        path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
        state = RunFile(project=project, generator_model=settings.groq_llm_model,
                        judge_model=judge_model, metrics=metrics, started=_now())
    state.thresholds = thresholds

    if service is None:
        from src.core.service import RAGService
        service = RAGService(project)
    scorers = metric_factory(metrics, judge, thresholds) if metrics else {}

    print(f"Results: {path}")
    stopped = None
    for i, case in enumerate(cases, start=1):
        existing = state.cases.get(case.id)
        try:
            if existing is None or existing.error is not None:
                result = ask_case(case, service)
                todo = metrics if case.kind == "answer" and result.error is None else []
            else:
                result, todo = existing, existing.missing_metrics(metrics)
                if not todo:
                    print(f"[{i}/{len(cases)}] {case.id}: already done")
                    continue
            state.cases[case.id] = result
            score_metrics(case, result, {m: scorers[m] for m in todo}, judge)
        except RateLimitStop as exc:
            stopped = str(exc)
            break
        finally:
            save_run(state, path)
        print(_progress(i, len(cases), result))

    selected = [state.cases[c.id] for c in cases if c.id in state.cases]
    summary = summarize(selected, state.metrics, thresholds,
                        sum(c.kind == "answer" for c in golden_cases))
    state.summary = summary
    save_run(state, path)
    markdown = path.with_suffix(".md")
    markdown.write_text(render_markdown(state, summary, path.stem), encoding="utf-8")
    print(format_summary(summary))
    print(f"Wrote {path} and {markdown}")
    if stopped:
        print(f"\nStopped early: Groq rate limit ({stopped}). Continue later with the same flags "
              f"plus:\n  --resume {path}", file=sys.stderr)
        return 2
    return 0 if summary.passed else 1


# ---------- CLI ----------

def parse_metrics(text: str) -> list[str]:
    names = [m.strip() for m in text.split(",") if m.strip()]
    unknown = [m for m in names if m not in METRIC_NAMES]
    if unknown:
        raise ValueError(f"unknown metric(s) {', '.join(unknown)}; choose from "
                         f"{', '.join(METRIC_NAMES)}")
    return [m for m in METRIC_NAMES if m in names]


def parse_thresholds(items: Sequence[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or name not in THRESHOLDS:
            raise ValueError(f"bad threshold '{item}': use NAME=VALUE with NAME one of "
                             f"{', '.join(THRESHOLDS)}")
        try:
            number = float(value)
        except ValueError:
            number = -1.0
        if not 0 <= number <= 1:
            raise ValueError(f"bad threshold '{item}': VALUE must be between 0 and 1")
        out[name] = number
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default="medrag")
    parser.add_argument("--limit", type=int, help="only the first N golden-set cases")
    parser.add_argument("--metrics", default=",".join(METRIC_NAMES),
                        help="comma-separated LLM metrics (default: all)")
    parser.add_argument("--no-llm-metrics", action="store_true",
                        help="no judge calls: retrieval hits and refusals only")
    parser.add_argument("--resume", type=Path, help="continue this results file")
    parser.add_argument("--judge-delay", type=float, default=JUDGE_DELAY_SECONDS,
                        help="seconds to wait after each judge call")
    parser.add_argument("--threshold", action="append", default=[], metavar="NAME=VALUE",
                        help=f"override a threshold ({', '.join(THRESHOLDS)})")
    parser.add_argument("--golden", type=Path, default=GOLDEN_SET)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        metrics = [] if args.no_llm_metrics else parse_metrics(args.metrics)
        thresholds = parse_thresholds(args.threshold)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return run(args.project, limit=args.limit, metrics=metrics, resume=args.resume,
                   judge_delay=args.judge_delay, thresholds=thresholds, golden=args.golden)
    except EvalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
