---
title: Meta DataClean Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Meta DataClean Env

Messy tables show up everywhere: exports from legacy tools, half-filled forms, spreadsheets someone “fixed” by hand. **Meta DataClean Env** is a small gym-style environment where an agent cleans synthetic tabular data **one step at a time**, through a **REST API**. It sits next to [OpenEnv](https://github.com/meta-pytorch/OpenEnv) in spirit: you get observations, you send structured actions, you read rewards. The heavy lifting is **pandas** plus **deterministic graders**, not a black-box judge.

---

## Why this idea

Most real analytics work is not choosing a fancier model. It is **getting the table into shape**: nulls, duplicates, weird formats, values that cannot possibly be true, keys that point nowhere. That work is repetitive, but it is not trivial—there are tradeoffs (drop vs impute, clip vs remove), and the “right” fix depends on context.

I wanted a place where an **agent** (or an LLM loop) could practice that loop: look at a snapshot, pick a concrete operation, see what changed, repeat. **RL-style interaction** fits because cleaning is naturally **sequential**: you rarely fix everything in one shot, and partial progress is measurable. A stepwise API forces the agent to commit to **discrete tools** instead of rewriting the whole CSV in one opaque blob, which keeps evaluation honest.

---

## Overview

Meta DataClean Env ships **six tasks** that ramp from “fill the nulls” to “everything is wrong at once.” Each episode **resets** a dataset (and optional multi-table setup), exposes a **short observation** (metadata + a few rows + grader messages), and accepts a **JSON action**. After every step the server recomputes a **score** in `[0, 1]` and a **reward** from how much that score moved.

The point is not to replace your ETL stack. It is a **controlled benchmark**: same seeds, same rules, same scoring—so you can compare policies, prompts, or models without arguing about what “clean” meant that day.

---

## How the system works (architecture)

Think of the server as a **state machine** with one hidden dataframe (or a pair of tables for the referential task). **`POST /reset`** picks a task, draws fresh messy data, seeds RNG, and returns the first observation. **`POST /step`** applies one action, runs the grader, updates step count, and returns `{ observation, reward, done, info }`.

A typical loop looks like this:

1. Client calls **`/reset`** with `task_id` and `seed`.
2. Client reads **observation** (what the agent is allowed to see).
3. Client chooses an action—**`inference.py`** is just one example: it wraps that choice as an LLM call that must output JSON matching `DataCleanAction`.
4. Client calls **`/step`** with that JSON.
5. Repeat until `done` is true or you hit your own step cap.

**REST role:** `/reset` and `/step` are the contract. Anything can sit on the client side: a scripted policy, reinforcement learning, or an LLM. **`inference.py`** is the “demo brain”: it builds a text prompt from the observation, hits a chat-completions endpoint, parses JSON, and forwards the action to `/step`. The environment never calls the model; **the client does**.

**Why a global `DataCleanEnvironment` in `server/app`?** FastAPI keeps one process-wide instance so the implementation stays small: no session IDs, no store, no locking. That is fine for **single-user** demos, local dev, and Spaces where you expect one conversation at a time.

**Limitation:** concurrent users would stomp on the same in-memory state. A production-shaped version would attach an **episode id** to each client (or user) and either spin **per-session env objects** in memory or persist state in **Redis** (or similar) with the id as key. **Multi-worker** Uvicorn would need the same fix—workers do not share RAM—so you would move episode state out-of-process or pin a user to one worker.

---

## Tasks

| # | Task ID | Difficulty | Max steps | What breaks |
|---|---------|------------|-----------|-------------|
| 1 | `missing_values` | Easy | 15 | Nulls in numeric and categorical columns |
| 2 | `duplicate_removal` | Easy / medium | 15 | Exact duplicates and casing variants on the same logical row |
| 3 | `type_format_fix` | Medium | 20 | Dates, phones, booleans, currency strings that disagree on format |
| 4 | `outlier_treatment` | Medium / hard | 20 | Extreme values; grader expects IQR-style treatment |
| 5 | `referential_integrity` | Hard | 25 | Two tables: orphan keys and inconsistent categorical labels |
| 6 | `full_pipeline_repair` | Hard | 35 | Duplicates, nulls, outliers, casing, booleans-as-strings, dates-as-strings—combined |

`max_steps` is enforced inside the environment: run out of steps without reaching a perfect score and the episode ends with an extra penalty (see rewards below).

---

## Action space

Each action is a **`DataCleanAction`** (Pydantic) the API can validate before anything touches pandas:

```python
class DataCleanAction(BaseModel):
    action_type: str
    column: Optional[str] = None
    params: Optional[dict] = {}
```

| `action_type` | Typical `params` | What it does |
|---------------|------------------|----------------|
| `fill_missing` | `{"strategy": "mean\|median\|mode\|ffill\|value"}`, optional `"value"` for `value` | Fills nulls in the chosen column |
| `drop_duplicates` | `{"subset": ["col", ...]}` or omit | Drops duplicate rows; multi-table task can pass `"table"` |
| `cast_type` | `{"dtype": "datetime\|float\|int\|bool\|str"}` | Coerces column type (with sensible maps for booleans) |
| `standardize_format` | `{"format": "currency_to_float\|phone_digits_only\|uppercase\|lowercase\|title_case\|strip_whitespace"}` | String cleanup helpers |
| `cap_outliers` | optional `lower` / `upper` | Clips numeric column using IQR unless bounds overridden |
| `drop_outliers` | `{}` | Keeps rows inside IQR bounds for that column |
| `rename_value` | `{"mapping": {"old": "new", ...}}` | Category cleanup |
| `drop_column` | `{}` (uses `column`) | Drops a column from the active frame |
| `fix_cross_column` | `{"ref_col", "ref_table", "target_table"}` | Removes orphan rows in `target_table` using keys from `ref_table` |
| `submit` | `{}` | Declares done; succeeds only if the grader already says score is 1.0 |

If something fails (unknown column, bad params), the observation still updates, and `info` may carry an `action_error` string so the client can log or retry.

---

## Observation space

```python
class DataCleanObservation(BaseModel):
    task_id: str
    task_description: str
    columns: List[ColumnInfo]
    data_sample: List[dict]
    issues_remaining: List[str]
    step_count: int
    last_action_result: str
```

**`columns`** carries name, dtype, null count, unique count, and a few sample values—enough to plan without sending the full grid. **`data_sample`** is capped (five rows) so payloads stay small on Spaces. **`issues_remaining`** is plain language from the grader: useful for LLM prompts and for humans watching a run. **`last_action_result`** is the environment’s short narrative of what just happened (fill counts, cast result, etc.).

For **`referential_integrity`**, the observation’s column list and sample are based on the **`orders`** table so the response size stays bounded; fixing the other table still happens through actions that know the multi-table context.

---

## How grading works

**Score** is a number between **0** and **1**. It is not a vibe check: every task has a **`_grade_*`** function that inspects the **current** tables after your last action and returns `(score, issues)`.

On **`/reset`**, the code generates both a **messy working copy** and **ground truth** used only inside the server. You never download the golden CSV. Some tasks compare directly to that reference (e.g. duplicate task row counts); others check **properties** you must satisfy (parsed dates, allowed tier labels, no orphan foreign keys). **Deterministic** means the same seed and the same sequence of actions yields the same score—important if you are comparing models or replaying a bug.

**Each action can move the score** when it actually fixes something the grader measures. Useless actions leave the score flat; destructive ones can drop it depending on the task. That is why “fair evaluation” is possible: everyone sees the same observation contract and the same scoring code in `server/dataclean_environment.py`.

---

## Reward function (exact rules)

Per step, after the action is applied:

1. `reward = (score_new - score_prev) * 2.0 - 0.01`
2. If `score_new >= 1.0`: add **`+0.5`**, episode **`done`**
3. If `step_count >= max_steps`: episode **`done`**; if score still `< 1.0`, subtract **`0.1`**
4. Clip the final reward to **`[-1.0, 1.0]`**

So the written formula is:

**`reward = (Score_new - Score_prev) * 2 - 0.01`**, plus the completion and timeout adjustments above.

---

## Reward logic (intuition)

The **`(Score_new - Score_prev) * 2`** term is doing the important work: it pays for **progress**, not for sitting on a high score. If you are already at 0.9 and you idle, the delta is zero—you stop earning. That nudges policies toward **actions that actually move the rubric**.

The **`-0.01`** per step is a small friction tax. Without it, an agent could spam neutral moves and treat the episode as free exploration. Here, pointless steps slowly bleed value unless they eventually help.

When you **clear the task** (`score >= 1.0`), the **`+0.5`** bump marks a clean finish. Together with the delta term, random flailing rarely looks attractive: you pay a little each step, big random jumps in score are uncommon if your actions are unrelated to the issues, and sustained improvement is what shows up as a positive return.

---

## Design decisions

**JSON actions** keep the boundary sharp: the model (or policy) must name a **verb** and **arguments** the server understands. That is easier to log, validate, and unit-test than free-form SQL or Python from an LLM.

**Why an LLM in `inference.py`?** It is a flexible baseline—you can swap the model or the prompt without forking the environment. The environment itself stays **non-learning** and deterministic; the intelligence lives in the client.

**Task ladder** isolates skills before stacking them. Debugging “the model fails everything at once” is miserable; debugging “it only mishandles outliers” is tractable.

**Observation = metadata + sample + issues** mirrors how people skim data in practice: schema first, a few rows, then a punch list of what still looks wrong. Sending the full table would hide the point (and blow the Space timeout).

---

## Limitations

- **`inference.py` depends on the model behaving**: bad JSON, wrong column names, or repeated `submit` will waste steps. The parser falls back to safe defaults sometimes; that can look like “the agent gave up” when it is really “the parse failed.”
- **Scale**: everything is in-memory pandas on modest row counts. This is not tuned for million-row files or streaming ETL.
- **Single global environment** per server process: not safe for many simultaneous clients without redesign (session ids, external store, or sticky sessions).
- **Synthetic data**: patterns are stylized on purpose. Transfer to your production schema still takes real integration work.
- **Grader ≠ every real-world policy**: some business rules you care about in production are not modeled here.

---

## Setup & usage

```bash
git clone <https://github.com/Jaswanthj006/CleanFlow-AI>
cd <MetaFinal>
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t metafinal .
docker run -p 7860:7860 metafinal
```

### Run the bundled LLM loop

Point it at a running server (same machine or Space), then:

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export ENV_BASE_URL=http://127.0.0.1:7860
python inference.py
```

If you prefer `API_KEY` over `HF_TOKEN`, that works too—the script checks both.

---

## API endpoints

| Method | Path | What it returns |
|--------|------|-----------------|
| `GET` | `/health` | Quick status JSON |
| `POST` | `/reset` | Body: `{ "task_id": "...", "seed": 42 }` → initial `DataCleanObservation` |
| `POST` | `/step` | Body: `DataCleanAction` → `{ observation, reward, done, info }` |
| `GET` | `/state` | Richer `DataCleanState`: totals, lists of issues, score, reward history |
| `GET` | `/tasks` | All task ids with `difficulty` and `max_steps` |
| `POST` | `/close` | Clears in-memory tables (handy if you want a clean process without restarting uvicorn) |

---

## Baseline performance scores

Rough numbers from an early LLM-driven run (your mileage will vary with model and prompt):

| Task | Score | Steps used |
|------|-------|------------|
| `missing_values` | 1.00 | 4 |
| `duplicate_removal` | 0.90 | 6 |
| `type_format_fix` | 0.85 | 8 |
| `outlier_treatment` | 0.80 | 7 |
| `referential_integrity` | 0.75 | 10 |
| `full_pipeline_repair` | 0.65 | 18 |

---

## Project structure

```
.
├── inference.py              # Example client / “agent brain”
├── models.py                 # Pydantic contracts shared by API + env
├── requirements.txt
├── Dockerfile
├── README.md
├── pyproject.toml
├── openenv.yaml
├── uv.lock
├── __init__.py
└── server/
    ├── __init__.py
    ├── app.py                # FastAPI routes
    └── dataclean_environment.py  # Tasks, actions, grading, rewards
```

---

Built as a coursework-style environment: small enough to read in one sitting, strict enough to score fairly, and open enough that you can rip out `inference.py` and plug in whatever agent you like.
