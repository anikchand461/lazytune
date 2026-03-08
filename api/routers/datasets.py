from fastapi import APIRouter, UploadFile, File
import shutil
import os

router = APIRouter(prefix="/datasets")

DATASET_PATH = "datasets/data.csv"


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):

    os.makedirs("datasets", exist_ok=True)

    with open(DATASET_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Dataset uploaded successfully"}
