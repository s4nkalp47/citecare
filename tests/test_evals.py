"""Phase 10: evals scoring, results file, resume and the Groq judge (no Groq, Qdrant or judge calls)."""

import json
import re
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from evals import judge as judge_module
from evals import run_evals
from evals.judge import GroqJudge, JudgeUsage
from evals.run_evals import CaseResult, EvalError, GoldenCase, MetricResult
from src.config import Settings
from src.core.schemas import Citation, RAGResponse
from tests.conftest import FakeGroqClient

NG136 = "NICE NG136 Hypertension"
NG28 = "NICE NG28 Type2 Diabetes"


def _node(title: str, text: str = "chunk", rec_ids: list[str] | None = None) -> NodeWithScore:
    metadata = {"title": title}
    if rec_ids is not None:
        metadata["rec_ids"] = rec_ids
    return NodeWithScore(node=TextNode(text=text, metadata=metadata), score=0.5)


def _response(answer="Offer an ARB [1].", confidence="high", titles=(NG136,), contexts=("ctx",)):
    return RAGResponse(answer=answer, confidence=confidence, contexts=list(contexts),
                       citations=[Citation(ref=i, title=t, source_type="guideline", snippet="s")
                                  for i, t in enumerate(titles, start=1)])


def _case(**overrides) -> GoldenCase:
    fields = {"id": "c1", "question": "q?", "expected_answer": "a", "expected_source": NG136,
              "recs": ["1.4.33"], "style": "guideline"} | overrides
    return GoldenCase.model_validate(fields)


# ---------- Golden set ----------

_FUNCTION_WORDS = {"a", "an", "the", "is", "are", "be", "for", "with", "to", "of", "and", "in",
                   "or", "what", "how", "which", "should", "when", "do", "i", "my", "it", "if"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _FUNCTION_WORDS


def test_golden_set_is_valid_and_balanced():
    cases = run_evals.load_golden_set()
    answers = [c for c in cases if c.kind == "answer"]
    negatives = [c for c in cases if c.kind == "refusal"]

    assert 18 <= len(answers) <= 24 and 3 <= len(negatives) <= 4
    assert {c.expected_source for c in answers} == {NG136, NG28, "NICE NG250 Pneumonia"}
    assert {c.style for c in answers} == {"guideline", "clinician", "two_recs"}
    assert all(c.style == "negative" and not c.recs for c in negatives)
    assert all(len(c.recs) == 2 for c in answers if c.style == "two_recs")
    # Round-robin order: --limit 3 covers all three guidelines, --limit 4 includes a negative.
    assert len({c.expected_source for c in cases[:3]}) == 3 and cases[3].kind == "refusal"


def test_golden_set_does_not_reuse_tuning_questions():
    from scripts.retrieval_check import CASES, HARD_NEGATIVES

    tuning = [c.question for c in CASES] + HARD_NEGATIVES
    for case in run_evals.load_golden_set():
        for question in tuning:
            a, b = _words(case.question), _words(question)
            assert a != b, case.id
            assert len(a & b) / len(a | b) < 0.5, (case.id, question)  # content-word Jaccard


def test_golden_case_validation(tmp_path):
    with pytest.raises(ValueError, match="expected_source"):
        _case(expected_source=None)
    assert _case().guideline == "NG136"
    assert _case(kind="refusal", expected_source=None, recs=[], style="negative").guideline is None

    path = tmp_path / "golden.json"
    path.write_text(json.dumps([_case().model_dump(), _case().model_dump()]), encoding="utf-8")
    with pytest.raises(EvalError, match="duplicate"):
        run_evals.load_golden_set(path)


# ---------- Scoring without an LLM ----------

def test_rec_hit_needs_every_rec_from_the_right_guideline():
    case = _case(recs=["1.4.41", "1.4.45"])
    both = [_node(NG136, rec_ids=["1.4.41"]), _node(NG28, rec_ids=["1.4.45"]),
            _node(NG136, rec_ids=["1.4.44", "1.4.45"])]
    assert run_evals.rec_hit(case, both)
    assert not run_evals.rec_hit(case, both[:2])  # 1.4.45 only from the wrong guideline
    assert not run_evals.rec_hit(case, [])


def test_rec_hit_text_fallback_and_title_fallback():
    page_chunk = _node(NG136, text="Step 2\n1.4.41 If hypertension is not controlled...")
    assert run_evals.rec_hit(_case(recs=["1.4.41"]), [page_chunk])
    assert not run_evals.rec_hit(_case(recs=["1.4.4"]), [page_chunk])  # whole number only

    table = _case(recs=[])  # table answers: any chunk from the expected source
    assert run_evals.rec_hit(table, [_node(NG28), _node(NG136.upper())])
    assert not run_evals.rec_hit(table, [_node(NG28)])


def test_title_hit_is_case_insensitive_on_citations():
    assert run_evals.title_hit(_case(), _response(titles=(NG28, NG136.lower())))
    assert not run_evals.title_hit(_case(), _response(titles=(NG28,)))
    assert not run_evals.title_hit(_case(), _response(titles=()))


@pytest.mark.parametrize("answer, confidence, refused", [
    ("No sufficiently relevant sources were found.", "insufficient_evidence", True),
    ("These guidelines don't cover pregnancy; see NICE's guideline on hypertension in pregnancy.",
     "low", True),
    ("This is outside the scope of the provided guidelines.", "medium", True),
    ("See NICE's guideline on type 1 diabetes; typically start 0.5 units/kg/day.", "low", False),
    ("Offer labetalol 100 mg twice daily.", "medium", False),
    ("Offer an ACE inhibitor or an ARB [1].", "high", False),
])
def test_is_refusal(answer, confidence, refused):
    assert run_evals.is_refusal(_response(answer=answer, confidence=confidence)) is refused


# ---------- Running: fakes ----------

class FakeService:
    """`ask_with_nodes` per question: a (response, nodes) pair, or an exception to raise."""

    def __init__(self, replies: dict, on_ask=None):
        self.replies = replies
        self.asked: list[str] = []
        self.on_ask = on_ask

    def ask_with_nodes(self, question):
        self.asked.append(question)
        if self.on_ask:
            self.on_ask(question)
        reply = self.replies[question]
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeJudge:
    def __init__(self, name="judge-model"):
        self.name = name
        self.usage = JudgeUsage()

    def get_model_name(self):
        return self.name


class FakeMetric:
    """Scores from a list (an Exception is raised instead); each measure costs 100 judge tokens."""

    def __init__(self, name, judge, scores, threshold):
        self.name, self.judge, self.scores, self.threshold = name, judge, list(scores), threshold
        self.test_cases = []
        self.score = self.reason = None

    def measure(self, test_case, _show_indicator=True):
        self.test_cases.append(test_case)
        self.judge.usage.prompt += 80
        self.judge.usage.completion += 20
        self.judge.usage.calls += 1
        score = self.scores.pop(0)
        if isinstance(score, BaseException):
            raise score
        self.score, self.reason = score, f"{self.name} reason"
        return score

    def is_successful(self):
        return self.score >= self.threshold


class RateLimited(Exception):
    status_code = 429


def _factory(scores: dict, made: dict):
    def make(names, judge, thresholds):
        made.update({n: FakeMetric(n, judge, scores.get(n, [1.0] * 10), thresholds[n])
                     for n in names})
        return dict(made)
    return make


GOLDEN = [
    {"id": "a1", "question": "q1", "expected_answer": "e1", "expected_source": NG136,
     "recs": ["1.4.33"], "style": "clinician"},
    {"id": "n1", "kind": "refusal", "question": "qn", "expected_answer": "none",
     "style": "negative"},
    {"id": "a2", "question": "q2", "expected_answer": "e2", "expected_source": NG28,
     "recs": [], "style": "guideline"},
]


@pytest.fixture
def golden(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(GOLDEN), encoding="utf-8")
    return path


def _replies(**overrides):
    replies = {
        "q1": (_response(contexts=("ARB if ACE cough",)), [_node(NG136, rec_ids=["1.4.33"])]),
        "qn": (_response(answer="No sources.", confidence="insufficient_evidence", titles=(),
                         contexts=()), []),
        "q2": (_response(titles=(NG28,), contexts=("metformin",)), [_node(NG28)]),
    }
    return replies | overrides


def _run(golden, tmp_path, service, metrics=("faithfulness", "correctness"), scores=None,
         made=None, judge=None, **kwargs) -> int:
    return run_evals.run("medrag", metrics=metrics, golden=golden, out_dir=tmp_path / "results",
                         service=service, judge=judge or FakeJudge(),
                         metric_factory=_factory(scores or {}, {} if made is None else made),
                         **kwargs)


def _results(tmp_path) -> tuple:
    [path] = (tmp_path / "results").glob("*.json")
    return path, json.loads(path.read_text(encoding="utf-8"))


# ---------- Running ----------

def test_run_scores_cases_and_writes_results(golden, tmp_path):
    made: dict = {}
    service = FakeService(_replies())
    code = _run(golden, tmp_path, service, made=made)

    assert code == 0
    path, data = _results(tmp_path)
    a1, n1, a2 = (data["cases"][i] for i in ("a1", "n1", "a2"))
    assert a1["rec_hit"] is True and a1["title_hit"] is True
    assert a1["metrics"]["faithfulness"]["score"] == 1.0 and a1["contexts"] == ["ARB if ACE cough"]
    assert a1["judge_tokens"] == {"prompt": 160, "completion": 40, "calls": 2,
                                  "estimated_calls": 0}
    assert a2["rec_hit"] is True  # no recs: title fallback over the retrieved nodes
    assert n1["refused"] is True and n1["metrics"] == {}  # negatives: no judge calls
    # One LLMTestCase per answer case per metric, with the retrieved contexts.
    tc = made["correctness"].test_cases
    assert [t.input for t in tc] == ["q1", "q2"]
    assert tc[0].retrieval_context == ["ARB if ACE cough"] and tc[0].expected_output == "e1"

    summary = data["summary"]
    assert summary["rec_hit_rate"] == 1.0 and summary["refusal_rate"] == 1.0
    assert summary["judge_tokens"]["mean_per_case"] == 200
    assert summary["judge_tokens"]["projected_full_set"] == 400  # 2 answer cases in the golden set
    markdown = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Checks" in markdown and "| a1 | clinician | yes | yes | - | high |" in markdown
    assert "**PASS**" in markdown


def test_results_file_is_saved_after_every_case(golden, tmp_path):
    seen = []

    def on_ask(question):
        files = list((tmp_path / "results").glob("*.json"))
        if files:
            seen.append(sorted(json.loads(files[0].read_text(encoding="utf-8"))["cases"]))

    _run(golden, tmp_path, FakeService(_replies(), on_ask=on_ask), metrics=())
    assert seen == [["a1"], ["a1", "n1"]]


def test_metric_error_is_recorded_and_the_run_continues(golden, tmp_path):
    code = _run(golden, tmp_path, FakeService(_replies()),
                scores={"faithfulness": [ValueError("Evaluation LLM outputted an invalid JSON"),
                                         0.9]})
    _, data = _results(tmp_path)
    assert data["cases"]["a1"]["metrics"]["faithfulness"]["error"].startswith("ValueError")
    assert data["cases"]["a2"]["metrics"]["faithfulness"]["score"] == 0.9
    assert data["summary"]["metrics"]["faithfulness"] == {"mean": 0.9, "n": 1, "errors": 1}
    assert code == 0  # the mean of the scored cases still meets the threshold


def test_ask_failure_is_recorded_and_fails_the_run(golden, tmp_path):
    code = _run(golden, tmp_path, FakeService(_replies(q1=RuntimeError("Qdrant down"))))
    _, data = _results(tmp_path)
    assert data["cases"]["a1"]["error"] == "RuntimeError: Qdrant down"
    assert data["cases"]["a2"]["metrics"]["correctness"]["score"] == 1.0
    assert data["summary"]["failed_cases"] == 1
    assert code == 1


def test_nothing_retrieved_scores_zero_without_judge_calls(golden, tmp_path):
    made: dict = {}
    empty = (_response(answer="No sources.", confidence="insufficient_evidence", titles=(),
                       contexts=()), [])
    _run(golden, tmp_path, FakeService(_replies(q1=empty)), made=made)
    _, data = _results(tmp_path)
    a1 = data["cases"]["a1"]
    assert a1["rec_hit"] is False
    assert a1["metrics"]["faithfulness"] == {"score": 0.0, "success": False,
                                             "reason": run_evals.NO_CONTEXT_REASON, "error": None}
    assert [t.input for t in made["faithfulness"].test_cases] == ["q2"]


def test_below_threshold_fails(golden, tmp_path):
    code = _run(golden, tmp_path, FakeService(_replies()), scores={"correctness": [0.5, 0.6]})
    _, data = _results(tmp_path)
    check = data["summary"]["checks"]["correctness"]
    assert check["value"] == pytest.approx(0.55) and check["passed"] is False
    assert code == 1
    assert _run(golden, tmp_path / "again", FakeService(_replies()),
                scores={"correctness": [0.5, 0.6]}, thresholds={"correctness": 0.5}) == 0


def test_rate_limit_stops_then_resume_finishes(golden, tmp_path):
    service = FakeService(_replies())
    code = _run(golden, tmp_path, service,
                scores={"correctness": [1.0, RateLimited("try again in 2h")]})
    assert code == 2
    path, data = _results(tmp_path)
    assert set(data["cases"]) == {"a1", "n1", "a2"}
    assert data["cases"]["a2"]["metrics"]["faithfulness"]["score"] == 1.0
    assert "correctness" not in data["cases"]["a2"]["metrics"]  # saved without the failed one

    made: dict = {}
    resumed = FakeService(_replies())
    code = _run(golden, tmp_path, resumed, made=made, resume=path)
    assert code == 0
    assert resumed.asked == []  # a1, n1 done; a2 only needed correctness, on the stored answer
    assert made["faithfulness"].test_cases == []
    assert [t.input for t in made["correctness"].test_cases] == ["q2"]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cases"]["a2"]["metrics"]["correctness"]["score"] == 1.0
    assert data["cases"]["a2"]["judge_tokens"]["calls"] == 3  # 2 before the stop, 1 on resume


def test_rate_limit_while_asking_stops(golden, tmp_path):
    code = _run(golden, tmp_path, FakeService(_replies(qn=RateLimited("429"))))
    assert code == 2
    _, data = _results(tmp_path)
    assert set(data["cases"]) == {"a1"}


def test_resume_adds_a_metric_and_re_asks_failed_cases(golden, tmp_path):
    _run(golden, tmp_path, FakeService(_replies(q1=RuntimeError("Qdrant down"))),
         metrics=("faithfulness",))
    path, _ = _results(tmp_path)

    resumed = FakeService(_replies())
    code = _run(golden, tmp_path, resumed, metrics=("correctness",), resume=path)
    assert resumed.asked == ["q1"]  # only the failed case is asked again
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metrics"] == ["faithfulness", "correctness"]
    assert set(data["cases"]["a2"]["metrics"]) == {"faithfulness", "correctness"}
    assert set(data["cases"]["a1"]["metrics"]) == {"correctness"}
    assert code == 0


def test_resume_rejects_another_project_or_judge(golden, tmp_path):
    _run(golden, tmp_path, FakeService(_replies()))
    path, _ = _results(tmp_path)
    with pytest.raises(EvalError, match="project"):
        run_evals.run("other", metrics=(), golden=golden, resume=path,
                      service=FakeService(_replies()))
    with pytest.raises(EvalError, match="judged by judge-model"):
        _run(golden, tmp_path, FakeService(_replies()), judge=FakeJudge("other-judge"),
             resume=path)
    with pytest.raises(EvalError, match="can't resume"):
        _run(golden, tmp_path, FakeService(_replies()), resume=tmp_path / "missing.json")


def test_limit_and_no_llm_metrics(golden, tmp_path):
    service = FakeService(_replies())
    code = run_evals.run("medrag", limit=1, metrics=(), golden=golden,
                         out_dir=tmp_path / "results", service=service)
    assert code == 0 and service.asked == ["q1"]
    _, data = _results(tmp_path)
    assert data["summary"]["checks"]["refusal_rate"]["passed"] is None  # no negative in scope
    assert data["cases"]["a1"]["metrics"] == {}


# ---------- Summary ----------

def _result(id_, kind="answer", **fields) -> CaseResult:
    return CaseResult(id=id_, kind=kind, style=fields.pop("style", "guideline"), question="q",
                      **fields)


def test_summarize_checks_and_styles():
    results = [
        _result("a", rec_hit=True, title_hit=False, style="clinician",
                metrics={"correctness": MetricResult(score=0.6)}),
        _result("b", rec_hit=False, title_hit=True,
                metrics={"correctness": MetricResult(error="ValueError: bad JSON")}),
        _result("n", kind="refusal", style="negative", refused=False),
    ]
    summary = run_evals.summarize(results, ["correctness"], run_evals.THRESHOLDS, 20)

    assert summary.rec_hit_rate == 0.5 and summary.title_hit_rate == 0.5
    assert summary.checks["rec_hit_rate"].passed is False
    assert summary.checks["correctness"].passed is True  # 0.6 >= 0.6
    assert summary.checks["refusal_rate"].value == 0.0 and summary.passed is False
    assert summary.by_style["clinician"] == {"cases": 1, "rec_hit_rate": 1.0, "correctness": 0.6}
    assert summary.metrics["correctness"].errors == 1


def test_summarize_errors_only_fails_and_no_data_is_not_checked():
    results = [_result("a", rec_hit=True, metrics={"faithfulness": MetricResult(error="x")})]
    summary = run_evals.summarize(results, ["faithfulness"], run_evals.THRESHOLDS, 20)
    assert summary.checks["faithfulness"].passed is False
    assert summary.checks["refusal_rate"].passed is None
    assert summary.passed is False


def test_summarize_judge_tokens():
    results = [_result("a", rec_hit=True, judge_tokens={"prompt": 900, "completion": 100,
                                                       "calls": 4, "estimated_calls": 1}),
               _result("b", rec_hit=True, judge_tokens={"prompt": 2500, "completion": 500,
                                                       "calls": 4}),
               _result("c", rec_hit=True)]  # resumed without judge calls: not in the mean
    tokens = run_evals.summarize(results, [], run_evals.THRESHOLDS, 21).judge_tokens
    assert tokens == {"total": 4000, "cases": 2, "mean_per_case": 2000, "estimated_calls": 1,
                      "calls": 8, "projected_full_set": 42000}


def test_parse_metrics_and_thresholds():
    assert run_evals.parse_metrics("correctness, faithfulness") == ["faithfulness", "correctness"]
    assert run_evals.parse_metrics("") == []
    with pytest.raises(ValueError, match="unknown metric"):
        run_evals.parse_metrics("faithfulness,bleu")
    assert run_evals.parse_thresholds(["correctness=0.7", "rec_hit_rate = 0.9"]) == {
        "correctness": 0.7, "rec_hit_rate": 0.9}
    for bad in ("correctness", "bleu=0.5", "correctness=high", "correctness=1.5"):
        with pytest.raises(ValueError, match="bad threshold"):
            run_evals.parse_thresholds([bad])


def test_main_parses_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(run_evals, "run", lambda project, **kw: seen.update(project=project, **kw)
                        or 1)
    assert run_evals.main(["--limit", "3", "--metrics", "correctness", "--threshold",
                           "correctness=0.8", "--judge-delay", "0.5"]) == 1
    assert seen["limit"] == 3 and seen["metrics"] == ["correctness"]
    assert seen["thresholds"] == {"correctness": 0.8} and seen["judge_delay"] == 0.5
    run_evals.main(["--no-llm-metrics"])
    assert seen["metrics"] == []


# ---------- Groq judge ----------

MODEL = "qwen/qwen3.8-27b"


def _judge(replies, **kwargs) -> tuple[GroqJudge, FakeGroqClient]:
    client = FakeGroqClient({MODEL: replies})
    return GroqJudge(MODEL, client=client, delay=0, **kwargs), client


def test_judge_parses_schema_replies_with_think_and_fences():
    from deepeval.metrics.faithfulness.schema import Claims

    judge, client = _judge(['<think>hmm</think>\n```json\n{"claims": ["Offer an ARB."]}\n```',
                            "plain text"])
    assert judge.generate("extract claims", schema=Claims) == Claims(claims=["Offer an ARB."])
    assert judge.generate("say something") == "plain text"
    assert client.calls[0]["messages"] == [{"role": "user", "content": "extract claims"}]
    assert client.calls[0]["temperature"] == 0


def test_judge_returns_text_when_schema_does_not_match():
    from deepeval.metrics.faithfulness.schema import Claims

    judge, _ = _judge(['{"wrong": 1}'])
    assert judge.generate("p", schema=Claims) == '{"wrong": 1}'  # DeepEval parses and reports


def test_judge_model_and_reasoning_from_settings(monkeypatch):
    monkeypatch.setattr(judge_module, "get_settings", lambda: Settings(
        _env_file=None, groq_judge_model=MODEL, groq_judge_reasoning_effort="none"))
    client = FakeGroqClient({MODEL: ["a"]})
    judge = GroqJudge(client=client, delay=0)
    judge.generate("p")
    assert judge.get_model_name() == MODEL and client.calls[0]["model"] == MODEL
    assert client.calls[0]["reasoning_effort"] == "none"

    judge, client = _judge(["a"], reasoning_effort="")
    judge.generate("p")
    assert "reasoning_effort" not in client.calls[0]


def test_judge_waits_after_each_call(monkeypatch):
    sleeps = []
    monkeypatch.setattr(judge_module.time, "sleep", sleeps.append)
    client = FakeGroqClient({MODEL: ["a", "b"]})
    judge = GroqJudge(MODEL, client=client, delay=1.5)
    judge.generate("p")
    judge.generate("p")
    assert sleeps == [1.5, 1.5]


def test_judge_counts_usage_and_estimates_without_it():
    judge, client = _judge(["x" * 40])
    judge.generate("p" * 400)  # FakeGroqClient replies have no usage: ~4 chars per token
    assert (judge.usage.prompt, judge.usage.completion, judge.usage.estimated_calls) == (100, 10, 1)

    usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=80)
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                            usage=usage)
    client.chat.completions.create = lambda **kwargs: reply
    judge.generate("p")
    assert (judge.usage.prompt, judge.usage.completion, judge.usage.calls) == (1300, 90, 2)
    assert judge.usage.estimated_calls == 1


def test_judge_retries_rate_limits(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    judge, client = _judge([RateLimited("try again in 1s"), "ok"])
    assert judge.generate("p") == "ok"
    assert len(client.calls) == 2


# ---------- DeepEval metrics with the judge (scripted replies, no network) ----------

def test_make_metrics_builds_deepeval_metrics():
    judge, _ = _judge([])
    metrics = run_evals.make_metrics(list(run_evals.METRIC_NAMES), judge, run_evals.THRESHOLDS)
    assert list(metrics) == list(run_evals.METRIC_NAMES)
    assert all(m.model is judge and m.async_mode is False for m in metrics.values())
    assert metrics["correctness"].threshold == 0.6
    assert metrics["correctness"].evaluation_steps == run_evals.CORRECTNESS_STEPS


def test_faithfulness_and_correctness_run_through_the_judge():
    from deepeval.test_case import LLMTestCase

    # Every faithfulness call (truths, claims, verdicts) accepts this reply.
    faithful = ('{"truths": ["Offer an ARB if an ACE inhibitor causes cough."], '
                '"claims": ["Offer an ARB."], "verdicts": [{"verdict": "yes"}]}')
    judge, client = _judge([faithful] * 3 + ['{"score": 8, "reason": "Names the ARB."}'])
    metrics = run_evals.make_metrics(["faithfulness", "correctness"], judge, run_evals.THRESHOLDS)
    case = LLMTestCase(input="ACE cough?", actual_output="Offer an ARB [1].",
                       expected_output="Offer an ARB.",
                       retrieval_context=["Offer an ARB if an ACE inhibitor causes cough."])

    metrics["faithfulness"].measure(case, _show_indicator=False)
    metrics["correctness"].measure(case, _show_indicator=False)

    assert metrics["faithfulness"].score == 1.0
    assert metrics["correctness"].score == pytest.approx(0.8)
    assert metrics["correctness"].is_successful()
    assert len(client.calls) == 4 and judge.usage.calls == 4
