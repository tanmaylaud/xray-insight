from fastapi import FastAPI, UploadFile, Form
from utils import predict as util_predict
from monitor import monitoring
import shutil
app = FastAPI()

@app.post("/predict/")
def predict(file:UploadFile, query:str = Form(), mode:str = Form()):
    with monitoring(mode=mode) as m:
        with open(file.filename, "wb") as f:
            shutil.copyfileobj(file.file, f)
        response = util_predict(file=file.filename, query=query)
        return {
            "response": response,
            "graph": m.graph
        }

