import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models import DataCleanAction, DataCleanObservation, DataCleanState, ColumnInfo


def _seed(seed: Optional[int]):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _gen_task1(seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _seed(seed)
    n = 50
    df = pd.DataFrame({
        "age":    np.random.randint(18, 70, n).astype(float),
        "salary": np.random.randint(30000, 120000, n).astype(float),
        "city":   np.random.choice(["Mumbai", "Delhi", "Bangalore", "Chennai"], n),
        "score":  np.random.uniform(0, 100, n),
    })
    for idx in random.sample(range(n), 8):
        df.at[idx, "age"] = np.nan
    for idx in random.sample(range(n), 6):
        df.at[idx, "salary"] = np.nan
    for idx in random.sample(range(n), 5):
        df.at[idx, "city"] = np.nan
    ground_truth = df.copy()
    ground_truth["age"] = ground_truth["age"].fillna(round(df["age"].mean(), 2))
    ground_truth["salary"] = ground_truth["salary"].fillna(round(df["salary"].median(), 2))
    ground_truth["city"] = ground_truth["city"].fillna(df["city"].mode()[0])
    return df, ground_truth


def _gen_task2(seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _seed(seed)
    base = pd.DataFrame({
        "customer_id": range(1, 31),
        "name":        [f"Customer_{i}" for i in range(1, 31)],
        "email":       [f"user{i}@example.com" for i in range(1, 31)],
        "amount":      np.random.randint(100, 5000, 30).astype(float),
    })
    exact_dups = base.iloc[[2, 7, 14]].copy()
    near_dups = base.iloc[[5, 10]].copy()
    near_dups["name"] = near_dups["name"].str.lower()
    near_dups["email"] = near_dups["email"].str.upper()
    df = pd.concat([base, exact_dups, near_dups], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, base.copy()


def _gen_task3(seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _seed(seed)
    n = 40
    dates_formats = ["2023-01-15", "15/01/2023", "Jan 15, 2023", "01-15-2023", "2023/01/15"]
    phones = ["+91-9876543210", "9876543210", "+91 98765 43210", "98765-43210", "(91)9876543210"]
    bools = ["Yes", "No", "1", "0", "True", "False", "yes", "no"]
    currencies = ["₹1,234.56", "1234.56", "Rs. 1234.56", "INR 1234.56", "1,234.56"]
    rows = []
    for i in range(n):
        rows.append({
            "join_date":   random.choice(dates_formats),
            "phone":       random.choice(phones),
            "is_active":   random.choice(bools),
            "revenue":     random.choice(currencies),
            "employee_id": str(random.randint(1000, 9999)),
        })
    df = pd.DataFrame(rows)
    ground_truth = pd.DataFrame({
        "join_date":   pd.to_datetime(["2023-01-15"] * n),
        "phone":       ["9876543210"] * n,
        "is_active":   [random.choice([True, False]) for _ in range(n)],
        "revenue":     [1234.56] * n,
        "employee_id": [str(random.randint(1000, 9999)) for _ in range(n)],
    })
    return df, ground_truth


def _gen_task4(seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _seed(seed)
    n = 60
    ages = list(np.random.randint(22, 58, n - 4).astype(float)) + [200.0, -5.0, 999.0, 150.0]
    salaries = list(np.random.randint(25000, 90000, n - 3).astype(float)) + [5000000.0, -10000.0, 9999999.0]
    scores = list(np.random.uniform(40, 95, n - 2)) + [500.0, -100.0]
    random.shuffle(ages)
    random.shuffle(salaries)
    random.shuffle(scores)
    df = pd.DataFrame({"age": ages, "salary": salaries, "score": scores})

    def cap_iqr(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        return series.clip(lower=q1 - 1.5 * (q3 - q1), upper=q3 + 1.5 * (q3 - q1))

    ground_truth = df.copy()
    for col in ["age", "salary", "score"]:
        ground_truth[col] = cap_iqr(df[col])
    return df, ground_truth


def _gen_task5(seed=42) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    _seed(seed)
    customers = pd.DataFrame({
        "customer_id": range(1, 21),
        "name":        [f"Customer_{i}" for i in range(1, 21)],
        "tier":        np.random.choice(["Gold", "Silver", "Bronze"], 20),
    })
    orders = pd.DataFrame({
        "order_id":    range(1001, 1041),
        "customer_id": list(range(1, 21)) * 2,
        "amount":      np.random.randint(100, 5000, 40).astype(float),
        "tier":        np.random.choice(["gold", "SILVER", "bronze", "Platinum"], 40),
    })
    orders.at[0, "customer_id"] = 999
    orders.at[5, "customer_id"] = 888
    orders.at[10, "customer_id"] = 777
    gt_orders = orders.copy()
    valid_ids = set(customers["customer_id"])
    gt_orders = gt_orders[gt_orders["customer_id"].isin(valid_ids)].reset_index(drop=True)
    tier_map = {"gold": "Gold", "silver": "Silver", "bronze": "Bronze", "SILVER": "Silver", "Platinum": "Bronze"}
    gt_orders["tier"] = gt_orders["tier"].map(lambda x: tier_map.get(x, x))
    return {"customers": customers, "orders": orders}, {"customers": customers, "orders": gt_orders}


def _gen_task6(seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _seed(seed)
    n = 60
    date_formats = ["2023-05-%02d" % i for i in range(1, 31)]
    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(n)],
        "age":         list(np.random.randint(22, 55, n - 3).astype(float)) + [999.0, -1.0, 200.0],
        "salary":      list(np.random.randint(30000, 100000, n - 2).astype(float)) + [np.nan, np.nan],
        "department":  np.random.choice(["Engineering", "Sales", "HR", "engineering", "SALES"], n),
        "join_date":   [random.choice(date_formats) for _ in range(n)],
        "is_manager":  np.random.choice(["Yes", "No", "1", "0", "True"], n),
        "performance": list(np.random.uniform(50, 95, n - 4)) + [200.0, -50.0, np.nan, np.nan],
    })
    dup_rows = df.iloc[[3, 7]].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    gt = df.drop_duplicates(subset=["employee_id"]).reset_index(drop=True)
    gt["salary"] = gt["salary"].fillna(gt["salary"].median())
    gt["performance"] = gt["performance"].fillna(gt["performance"].median())

    def cap_iqr(s):
        q1, q3 = s.dropna().quantile(0.25), s.dropna().quantile(0.75)
        return s.clip(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))

    gt["age"] = cap_iqr(gt["age"])
    gt["performance"] = cap_iqr(gt["performance"])
    gt["department"] = gt["department"].map(lambda x: {"engineering": "Engineering", "SALES": "Sales"}.get(x, x))
    gt["is_manager"] = gt["is_manager"].map({"Yes": True, "No": False, "1": True, "0": False, "True": True})
    gt["join_date"] = pd.to_datetime(gt["join_date"])
    return df, gt


def _grade_task1(current, ground_truth):
    issues, score = [], 0.0
    for col in ["age", "salary", "city"]:
        null_count = current[col].isnull().sum()
        if null_count == 0:
            score += 1.0 / 3
        else:
            issues.append(f"{col} still has {null_count} missing values")
    return round(score, 2), issues


def _grade_task2(current, ground_truth):
    issues = []
    expected_rows = len(ground_truth)
    actual_rows = len(current)
    exact_dups = current.duplicated(subset=["customer_id"]).sum()
    row_score = 1.0 if actual_rows == expected_rows else max(0, 1 - abs(actual_rows - expected_rows) / expected_rows)
    if exact_dups > 0:
        issues.append(f"{exact_dups} duplicate customer_id rows remain")
    if actual_rows != expected_rows:
        issues.append(f"Expected {expected_rows} rows, got {actual_rows}")
    return round(row_score * (1.0 - min(1.0, exact_dups / 5.0)), 2), issues


def _grade_task3(current, ground_truth):
    issues, score = [], 0.0
    checks = {
        "join_date": lambda s: pd.to_datetime(s, errors="coerce").notna().all(),
        "is_active": lambda s: s.dropna().isin([True, False, 0, 1]).all(),
        "revenue":   lambda s: pd.to_numeric(s, errors="coerce").notna().all(),
    }
    for col, check_fn in checks.items():
        if col in current.columns:
            try:
                if check_fn(current[col]):
                    score += 1.0 / len(checks)
                else:
                    issues.append(f"{col} not properly standardized")
            except Exception:
                issues.append(f"{col} check failed")
    return round(score, 2), issues


def _grade_task4(current, ground_truth):
    issues, score = [], 0.0
    for col in ["age", "salary", "score"]:
        q1, q3 = ground_truth[col].quantile(0.25), ground_truth[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((current[col] < q1 - 1.5 * iqr) | (current[col] > q3 + 1.5 * iqr)).sum()
        if outliers == 0:
            score += 1.0 / 3
        else:
            issues.append(f"{col} still has {outliers} outliers")
    return round(score, 2), issues


def _grade_task5(current, ground_truth):
    issues, score = [], 0.0
    orders = current.get("orders", pd.DataFrame())
    customers = current.get("customers", pd.DataFrame())
    if orders.empty or customers.empty:
        return 0.0, ["Missing tables"]
    orphaned = orders[~orders["customer_id"].isin(set(customers["customer_id"]))]
    if len(orphaned) == 0:
        score += 0.5
    else:
        issues.append(f"{len(orphaned)} orphaned order records remain")
    bad_tiers = orders[~orders["tier"].isin({"Gold", "Silver", "Bronze"})]
    if len(bad_tiers) == 0:
        score += 0.5
    else:
        issues.append(f"{len(bad_tiers)} orders have invalid tier values")
    return round(score, 2), issues


def _grade_task6(current, ground_truth):
    issues, score = [], 0.0
    weights = {"duplicates": 0.2, "missing": 0.2, "outliers": 0.2, "dept_case": 0.2, "bool_type": 0.1, "date_type": 0.1}

    if current.duplicated(subset=["employee_id"]).sum() == 0:
        score += weights["duplicates"]
    else:
        issues.append(f"{current.duplicated(subset=['employee_id']).sum()} duplicate employee_id rows remain")

    null_total = current[["salary", "performance"]].isnull().sum().sum()
    if null_total == 0:
        score += weights["missing"]
    else:
        issues.append(f"{null_total} missing values in salary/performance")

    q1, q3 = ground_truth["age"].quantile(0.25), ground_truth["age"].quantile(0.75)
    iqr = q3 - q1
    outliers = ((current["age"] < q1 - 1.5 * iqr) | (current["age"] > q3 + 1.5 * iqr)).sum()
    if outliers == 0:
        score += weights["outliers"]
    else:
        issues.append(f"{outliers} age outliers remain")

    bad_dept = current[~current["department"].isin(["Engineering", "Sales", "HR"])].shape[0]
    if bad_dept == 0:
        score += weights["dept_case"]
    else:
        issues.append(f"{bad_dept} rows with non-standard department values")

    if current["is_manager"].dtype == bool or current["is_manager"].isin([True, False]).all():
        score += weights["bool_type"]
    else:
        issues.append("is_manager column not converted to boolean")

    try:
        pd.to_datetime(current["join_date"], errors="raise")
        score += weights["date_type"]
    except Exception:
        issues.append("join_date not converted to datetime")

    return round(score, 2), issues


TASKS = {
    "missing_values":        {"difficulty": "easy",        "max_steps": 15},
    "duplicate_removal":     {"difficulty": "easy_medium", "max_steps": 15},
    "type_format_fix":       {"difficulty": "medium",      "max_steps": 20},
    "outlier_treatment":     {"difficulty": "medium_hard", "max_steps": 20},
    "referential_integrity": {"difficulty": "hard",        "max_steps": 25},
    "full_pipeline_repair":  {"difficulty": "hard",        "max_steps": 35},
}

TASK_DESCRIPTIONS = {
    "missing_values": (
        "This dataset has missing values in the 'age' (numeric), 'salary' (numeric), "
        "and 'city' (categorical) columns. Fill missing numeric values with the column mean "
        "or median and missing categorical values with the mode."
    ),
    "duplicate_removal": (
        "This customer dataset contains exact duplicate rows and near-duplicate rows "
        "(same data with different casing). Remove all duplicates keeping only the first "
        "occurrence of each unique customer."
    ),
    "type_format_fix": (
        "This dataset has columns with inconsistent formats: dates in 5 different formats, "
        "phone numbers with mixed separators, boolean values stored as Yes/No/1/0/True, "
        "and currency values with symbols. Standardize all columns to clean, consistent types."
    ),
    "outlier_treatment": (
        "This dataset contains statistical outliers in 'age', 'salary', and 'score' columns "
        "(e.g., age=200, salary=5000000). Detect outliers using IQR method and cap them "
        "at the IQR bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)."
    ),
    "referential_integrity": (
        "You have two tables: 'customers' and 'orders'. Some orders reference customer_ids "
        "that don't exist in the customers table (orphaned records). Also, the 'tier' column "
        "in orders has inconsistent casing (gold/SILVER/Platinum). Fix both issues."
    ),
    "full_pipeline_repair": (
        "This dataset has ALL types of issues: duplicate employee records, missing salary and "
        "performance values, age outliers, inconsistent department casing (engineering/SALES), "
        "boolean is_manager stored as strings, and join_date as strings. Fix everything."
    ),
}


class DataCleanEnvironment:

    def __init__(self):
        self._task_id: str = "missing_values"
        self._df: Optional[pd.DataFrame] = None
        self._tables: Optional[Dict[str, pd.DataFrame]] = None
        self._ground_truth = None
        self._step_count: int = 0
        self._max_steps: int = 15
        self._done: bool = False
        self._episode_rewards: List[float] = []
        self._issues_remaining: List[str] = []
        self._total_issues: int = 0
        self._issues_fixed: int = 0

    def reset(self, task_id: str = "missing_values", seed: int = 42) -> DataCleanObservation:
        self._task_id = task_id
        self._step_count = 0
        self._done = False
        self._episode_rewards = []
        self._max_steps = TASKS[task_id]["max_steps"]

        if task_id == "missing_values":
            self._df, self._ground_truth = _gen_task1(seed)
            self._tables = None
        elif task_id == "duplicate_removal":
            self._df, self._ground_truth = _gen_task2(seed)
            self._tables = None
        elif task_id == "type_format_fix":
            self._df, self._ground_truth = _gen_task3(seed)
            self._tables = None
        elif task_id == "outlier_treatment":
            self._df, self._ground_truth = _gen_task4(seed)
            self._tables = None
        elif task_id == "referential_integrity":
            self._tables, self._ground_truth = _gen_task5(seed)
            self._df = None
        elif task_id == "full_pipeline_repair":
            self._df, self._ground_truth = _gen_task6(seed)
            self._tables = None

        score, issues = self._grade()
        self._issues_remaining = issues
        self._total_issues = max(len(issues), 1)
        self._issues_fixed = 0
        return self._make_observation("Episode started. Analyze the data and begin cleaning.")

    def step(self, action: DataCleanAction) -> Tuple[DataCleanObservation, float, bool, Dict]:
        if self._done:
            return self._make_observation("Episode already done."), 0.0, True, {"error": "Episode is done. Call reset()."}

        self._step_count += 1
        prev_score, _ = self._grade()
        result_msg, action_error = self._apply_action(action)
        new_score, issues = self._grade()
        self._issues_remaining = issues
        self._issues_fixed = self._total_issues - len(issues)

        reward = (new_score - prev_score) * 2.0 - 0.01
        if new_score >= 1.0:
            reward += 0.5
            self._done = True
        if self._step_count >= self._max_steps:
            self._done = True
            if new_score < 1.0:
                reward -= 0.1

        reward = round(max(-1.0, min(1.0, reward)), 4)
        self._episode_rewards.append(reward)
        obs = self._make_observation(result_msg)
        return obs, reward, self._done, {"score": new_score, "issues_fixed": self._issues_fixed, "issues_remaining": len(issues), "action_error": action_error}

    def state(self) -> DataCleanState:
        score, _ = self._grade()
        return DataCleanState(
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            total_issues=self._total_issues,
            issues_fixed=self._issues_fixed,
            issues_remaining=self._issues_remaining,
            step_count=self._step_count,
            max_steps=self._max_steps,
            done=self._done,
            current_score=score,
            episode_rewards=self._episode_rewards,
        )

    def close(self):
        self._df = None
        self._tables = None
        self._ground_truth = None

    def _get_active_df(self):
        if self._task_id == "referential_integrity":
            return self._tables.get("orders") if self._tables else None
        return self._df

    def _grade(self):
        if self._task_id == "missing_values":
            return _grade_task1(self._df, self._ground_truth)
        elif self._task_id == "duplicate_removal":
            return _grade_task2(self._df, self._ground_truth)
        elif self._task_id == "type_format_fix":
            return _grade_task3(self._df, self._ground_truth)
        elif self._task_id == "outlier_treatment":
            return _grade_task4(self._df, self._ground_truth)
        elif self._task_id == "referential_integrity":
            return _grade_task5(self._tables, self._ground_truth)
        elif self._task_id == "full_pipeline_repair":
            return _grade_task6(self._df, self._ground_truth)
        return 0.0, ["Unknown task"]

    def _apply_action(self, action: DataCleanAction):
        at = action.action_type
        col = action.column
        params = action.params or {}
        try:
            if at == "fill_missing":          return self._act_fill_missing(col, params)
            elif at == "drop_duplicates":     return self._act_drop_duplicates(params)
            elif at == "cast_type":           return self._act_cast_type(col, params)
            elif at == "standardize_format":  return self._act_standardize_format(col, params)
            elif at == "cap_outliers":        return self._act_cap_outliers(col, params)
            elif at == "drop_outliers":       return self._act_drop_outliers(col, params)
            elif at == "rename_value":        return self._act_rename_value(col, params)
            elif at == "drop_column":         return self._act_drop_column(col)
            elif at == "fix_cross_column":    return self._act_fix_cross_column(params)
            elif at == "submit":
                score, issues = self._grade()
                if score >= 1.0:
                    self._done = True
                    return "Task submitted successfully. Score: 1.0", None
                return f"Submitted but score is {score:.2f}. Issues remain: {issues}", None
            else:
                return f"Unknown action type: {at}", f"Unknown action: {at}"
        except Exception as e:
            return f"Action failed: {str(e)}", str(e)

    def _act_fill_missing(self, col, params):
        df = self._get_active_df()
        if df is None or col not in df.columns:
            return f"Column '{col}' not found.", f"Column not found: {col}"
        strategy = params.get("strategy", "mean")
        null_before = df[col].isnull().sum()
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "value":
            df[col] = df[col].fillna(params.get("value", 0))
        else:
            return f"Unknown strategy: {strategy}", f"Unknown strategy: {strategy}"
        return f"Filled {null_before - df[col].isnull().sum()} nulls in '{col}' using {strategy}.", None

    def _act_drop_duplicates(self, params):
        if self._df is None:
            if self._tables:
                table = params.get("table", "orders")
                df = self._tables.get(table)
                before = len(df)
                self._tables[table] = df.drop_duplicates(subset=params.get("subset")).reset_index(drop=True)
                return f"Dropped {before - len(self._tables[table])} duplicates from {table}.", None
            return "No dataframe available.", "No dataframe"
        before = len(self._df)
        self._df = self._df.drop_duplicates(subset=params.get("subset")).reset_index(drop=True)
        return f"Dropped {before - len(self._df)} duplicate rows.", None

    def _act_cast_type(self, col, params):
        df = self._get_active_df()
        if df is None or col not in df.columns:
            return f"Column '{col}' not found.", f"Column not found: {col}"
        target = params.get("dtype", "str")
        if target == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif target == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif target == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif target == "bool":
            bool_map = {"yes": True, "no": False, "1": True, "0": False, "true": True, "false": False,
                        "Yes": True, "No": False, "True": True, "False": False, 1: True, 0: False}
            df[col] = df[col].map(bool_map)
        elif target == "str":
            df[col] = df[col].astype(str)
        else:
            return f"Unknown dtype: {target}", f"Unknown dtype: {target}"
        return f"Cast '{col}' to {target}.", None

    def _act_standardize_format(self, col, params):
        df = self._get_active_df()
        if df is None or col not in df.columns:
            return f"Column '{col}' not found.", f"Column not found: {col}"
        fmt = params.get("format", "")
        if fmt == "currency_to_float":
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[₹Rs.,INR\s]", "", regex=True), errors="coerce")
        elif fmt == "phone_digits_only":
            df[col] = df[col].astype(str).str.replace(r"[^\d]", "", regex=True).str[-10:]
        elif fmt == "uppercase":
            df[col] = df[col].str.upper()
        elif fmt == "lowercase":
            df[col] = df[col].str.lower()
        elif fmt == "title_case":
            df[col] = df[col].str.title()
        elif fmt == "strip_whitespace":
            df[col] = df[col].str.strip()
        else:
            return f"Unknown format: {fmt}", f"Unknown format: {fmt}"
        return f"Standardized '{col}' with format '{fmt}'.", None

    def _act_cap_outliers(self, col, params):
        df = self._get_active_df()
        if df is None or col not in df.columns:
            return f"Column '{col}' not found.", f"Column not found: {col}"
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower = params.get("lower", q1 - 1.5 * iqr)
        upper = params.get("upper", q3 + 1.5 * iqr)
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        return f"Capped {before} outliers in '{col}' to [{lower:.2f}, {upper:.2f}].", None

    def _act_drop_outliers(self, col, params):
        df = self._get_active_df()
        if df is None or col not in df.columns:
            return f"Column '{col}' not found.", f"Column not found: {col}"
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
        before = len(df)
        if self._task_id == "referential_integrity":
            self._tables["orders"] = df[mask].reset_index(drop=True)
            after_len = len(self._tables["orders"])
        else:
            self._df = df[mask].reset_index(drop=True)
            after_len = len(self._df)
        return f"Dropped {before - after_len} outlier rows based on '{col}'.", None

    def _act_rename_value(self, col, params):
        df = self._get_active_df()
        if df is None or col not in df.columns:
            return f"Column '{col}' not found.", f"Column not found: {col}"
        mapping = params.get("mapping", {})
        df[col] = df[col].replace(mapping)
        return f"Renamed values in '{col}': {mapping}.", None

    def _act_drop_column(self, col):
        df = self._get_active_df()
        if df is None or col not in df.columns:
            return f"Column '{col}' not found.", f"Column not found: {col}"
        if self._task_id == "referential_integrity":
            self._tables["orders"] = df.drop(columns=[col])
        else:
            self._df = df.drop(columns=[col])
        return f"Dropped column '{col}'.", None

    def _act_fix_cross_column(self, params):
        if self._tables is None:
            return "No multi-table context.", "No tables available"
        ref_col = params.get("ref_col", "customer_id")
        ref_table = params.get("ref_table", "customers")
        target_table = params.get("target_table", "orders")
        valid_ids = set(self._tables[ref_table][ref_col])
        orders = self._tables[target_table]
        before = len(orders)
        self._tables[target_table] = orders[orders[ref_col].isin(valid_ids)].reset_index(drop=True)
        return f"Removed {before - len(self._tables[target_table])} orphaned rows from '{target_table}'.", None

    def _make_observation(self, last_result: str) -> DataCleanObservation:
        score, issues = self._grade()
        df_view = self._tables.get("orders", pd.DataFrame()) if (self._task_id == "referential_integrity" and self._tables) else (self._df if self._df is not None else pd.DataFrame())
        columns = [
            ColumnInfo(
                name=c,
                dtype=str(df_view[c].dtype),
                null_count=int(df_view[c].isnull().sum()),
                unique_count=int(df_view[c].nunique()),
                sample_values=df_view[c].dropna().head(3).tolist(),
            )
            for c in df_view.columns
        ]
        return DataCleanObservation(
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            columns=columns,
            data_sample=df_view.head(5).fillna("NULL").to_dict(orient="records"),
            issues_remaining=issues,
            step_count=self._step_count,
            last_action_result=last_result,
        )
