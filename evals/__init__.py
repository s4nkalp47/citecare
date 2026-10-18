"""Phase 10: answer-quality evals (DeepEval with a Groq judge). See evals/run_evals.py."""

import os

# Set before anything imports deepeval: it would otherwise copy .env into os.environ (settings
# come only from src.config.get_settings) and send telemetry.
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

# Pause after every judge call: a metric makes several calls in a row, and the free tier limits
# requests and tokens per minute.
JUDGE_DELAY_SECONDS = 2.0
