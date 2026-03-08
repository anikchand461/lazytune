from pydantic import BaseModel
from typing import Dict

class OptimizeResponse(BaseModel):
    best_params: Dict
    score: float
    model: str
