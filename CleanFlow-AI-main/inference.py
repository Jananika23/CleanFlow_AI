import os
import json
from urllib import request
from urllib.error import URLError, HTTPError

from openai import OpenAI

# --- Required env vars ---
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "gpt-4.1-mini"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:7860"

TASKS = [
    "missing_values",
    "duplicate_removal",
    "type_format_fix",
    "outlier_treatment",
    "referential_integrity",
    "full_pipeline_repair",
]

MAX_STEPS = 30
BENCHMARK = "dataclean"


def _bool_lower(x: bool) -> str:
    return str(x).lower()


def _post_json(url: str, payload: dict) -> dict:
    try:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, ValueError, OSError):
        return {}


def llm_call(client, prompt):
    """Call LLM via OpenAI client."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return '{"action_type": "submit", "column": null, "params": {}}'


def build_prompt(obs):
    issues = obs.get("issues_remaining", [])
    sample = json.dumps(obs.get("data_sample", [])[:3], indent=2, default=str)
    columns = []
    for c in obs.get("columns", []):
        columns.append(
            f"  - {c.get('name','?')} (dtype={c.get('dtype','?')}, "
            f"nulls={c.get('null_count','?')}, uniques={c.get('unique_count','?')}, "
            f"samples={c.get('sample_values',[])})"
        )
    return f"""You are a data cleaning agent. Your task:
{obs.get('task_description', '')}

Current dataset columns:
{chr(10).join(columns)}

Data sample (first 3 rows):
{sample}

Issues still remaining: {issues}
Step: {obs.get('step_count', 0)}
Last action result: {obs.get('last_action_result', '')}

Choose the BEST next action to fix one of the remaining issues.
Respond with ONLY a valid JSON object:
{{
  "action_type": "<fill_missing|drop_duplicates|cast_type|standardize_format|cap_outliers|drop_outliers|rename_value|fix_cross_column|submit>",
  "column": "<column name or null>",
  "params": {{}}
}}

Action params guide:
- fill_missing: {{"strategy": "mean|median|mode|ffill|value"}}
- drop_duplicates: {{"subset": [cols] or null}}
- cast_type: {{"dtype": "datetime|float|int|bool|str"}}
- standardize_format: {{"format": "currency_to_float|phone_digits_only|uppercase|lowercase|title_case|strip_whitespace"}}
- cap_outliers / drop_outliers: {{}}
- rename_value: {{"mapping": {{"old": "new"}}}}
- fix_cross_column: {{"ref_col": "customer_id", "ref_table": "customers", "target_table": "orders"}}
- submit: {{}}

Respond with ONLY the JSON object."""


def parse_llm_action(text):
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if len(lines) > 2 else t
    try:
        action = json.loads(t)
        action.setdefault("action_type", "submit")
        action.setdefault("column", None)
        action.setdefault("params", {})
        return action
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', t, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {"action_type": "submit", "column": None, "params": {}}


def run_task(task_id, client):
    last_action_error = None
    rewards = []
    step_num = 0
    done = False
    success = False
    score = 0.0

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        obs = _post_json(f"{ENV_BASE_URL}/reset", {"task_id": task_id, "seed": 42})
        if not obs:
            raise ValueError("Empty reset response")

        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            prompt = build_prompt(obs)
            action_text = llm_call(client, prompt)
            action = parse_llm_action(action_text)

            try:
                result = _post_json(f"{ENV_BASE_URL}/step", action)
                if not result:
                    raise ValueError("Empty step response")

                obs    = result.get("observation", obs)
                reward = max(0.01, min(0.99, float(result.get("reward", 0.01))))
                done   = bool(result.get("done", False))
                info   = result.get("info", {})
                score  = max(0.01, min(0.99, float(info.get("score", score))))
                last_action_error = info.get("action_error") or None
                rewards.append(reward)

                action_str = (
                    f"{action['action_type']}("
                    f"col={action.get('column')!r},"
                    f"params={json.dumps(action.get('params', {}))})"
                )
                error_str = str(last_action_error) if last_action_error else "null"
                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={reward:.2f} done={_bool_lower(done)} "
                    f"error={error_str}",
                    flush=True,
                )

                if done:
                    success = score >= 1.0
                    break

            except Exception as e:
                last_action_error = str(e)
                rewards.append(0.01)
                print(
                    f"[STEP] step={step_num} action=error reward=0.00 "
                    f"done=false error={str(e)}",
                    flush=True,
                )

    except Exception as e:
        last_action_error = str(e)
        if step_num == 0:
            rewards.append(0.01)
            print(
                f"[STEP] step=1 action=error reward=0.00 "
                f"done=false error={str(e)}",
                flush=True,
            )
            step_num = 1

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"
    # Score must be strictly between 0 and 1 (not 0.0, not 1.0)
    clamped_score = max(0.01, min(0.99, score))
    print(
        f"[END] success={_bool_lower(success)} "
        f"steps={step_num} score={clamped_score:.2f} rewards={rewards_str}",
        flush=True,
    )


def main():
    # Create OpenAI client - wrapped in try/except so it never crashes the script
    client = None
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception:
        pass

    # If SDK fails, fall back to a minimal stub that uses raw urllib
    if client is None:
        class FallbackClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        import urllib.request as ur
                        url = API_BASE_URL.rstrip("/") + "/chat/completions"
                        body = json.dumps({
                            "model": kwargs.get("model", MODEL_NAME),
                            "messages": kwargs.get("messages", []),
                            "max_tokens": kwargs.get("max_tokens", 300),
                            "temperature": kwargs.get("temperature", 0.0),
                        }).encode()
                        req = ur.Request(url, data=body,
                            headers={"Authorization": f"Bearer {API_KEY}",
                                     "Content-Type": "application/json"})
                        with ur.urlopen(req, timeout=60) as r:
                            data = json.loads(r.read())
                        class Msg:
                            content = data["choices"][0]["message"]["content"]
                        class Choice:
                            message = Msg()
                        class Resp:
                            choices = [Choice()]
                        return Resp()
        client = FallbackClient()

    for task_id in TASKS:
        run_task(task_id, client)


if __name__ == "__main__":
    main()