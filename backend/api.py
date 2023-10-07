from fastapi import FastAPI

app = FastAPI()

@app.get("/predict")
def predict(query: str, image, mode, metadata):
    return {
        "response": "test",
        "graph": None
    }