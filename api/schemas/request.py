from pydantic import BaseModel
from typing import Dict


class OptimizeRequest(BaseModel):
    model: str
    target: str
    metric: str
    param_grid: Dict
