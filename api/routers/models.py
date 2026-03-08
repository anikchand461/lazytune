from fastapi import APIRouter

router = APIRouter(prefix="/models")

@router.get("/")
def get_models():
    return {
        "models": [
            "RandomForestClassifier",
            "SVC",
            "LogisticRegression",
            "RandomForestRegressor",
            "LinearRegression"
        ]
    }
