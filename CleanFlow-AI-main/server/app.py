import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from models import DataCleanAction, DataCleanObservation, DataCleanState
from server.dataclean_environment import DataCleanEnvironment, TASKS

app = FastAPI(title="DataClean Environment", version="1.0.0")
env = DataCleanEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "missing_values"
    seed: Optional[int] = 42


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "dataclean_env", "version": "1.0.0"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    if request is None:
        request = ResetRequest()
    task_id = request.task_id or "missing_values"
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: '{task_id}'. Available: {list(TASKS.keys())}"
        )
    return env.reset(task_id=task_id, seed=request.seed or 42).model_dump()


@app.post("/step")
async def step(action: DataCleanAction):
    try:
        obs, reward, done, info = env.step(action)
        return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    return env.state().model_dump()


@app.get("/tasks")
async def list_tasks():
    return {
        task_id: {"difficulty": meta["difficulty"], "max_steps": meta["max_steps"]}
        for task_id, meta in TASKS.items()
    }


@app.post("/close")
async def close():
    env.close()
    return {"status": "closed"}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", 7860)), reload=False)


if __name__ == "__main__":
    main()
