from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import optimize, datasets

app = FastAPI(title="LazyTune API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(optimize.router)


@app.get("/")
def root():
    return {"message": "LazyTune API is running"}
