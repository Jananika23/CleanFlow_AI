
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class DataCleanAction(BaseModel):
    action_type: str
    column: Optional[str] = None
    params: Optional[Dict[str, Any]] = {}


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any]


class DataCleanObservation(BaseModel):
    task_id: str
    task_description: str
    columns: List[ColumnInfo]
    data_sample: List[Dict[str, Any]]
    issues_remaining: List[str]
    step_count: int
    last_action_result: str


class DataCleanState(BaseModel):
    task_id: str
    task_description: str
    total_issues: int
    issues_fixed: int
    issues_remaining: List[str]
    step_count: int
    max_steps: int
    done: bool
    current_score: float
    episode_rewards: List[float]
