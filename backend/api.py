from fastapi import FastAPI, UploadFile
from utils import predict as util_predict
import shutil
app = FastAPI()

@app.post("/predict/")
def predict(file:UploadFile):
    print(file.file)
    with open(file.filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    response = util_predict(file=file.filename)
    return {
        "response": response,
        "graph": None
    }

