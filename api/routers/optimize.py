from fastapi import APIRouter
from api.schemas.request import OptimizeRequest
from api.services.tuning_service import run_tuning

router = APIRouter(prefix="/optimize")


@router.post("/")
def optimize(req: OptimizeRequest):

    dataset_path = "datasets/data.csv"

    best_params, score = run_tuning(
        dataset_path,
        req.model,
        req.target,
        req.metric,
        req.param_grid
    )

    return {
        "model": req.model,
        "best_params": best_params,
        "score": score
    }
