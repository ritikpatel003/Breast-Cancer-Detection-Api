import joblib
import sklearn
from fastapi import FastAPI

app = FastAPI()

model = joblib.load('model.joblib')


def prediction(model, data):
    res = int(model.predict([data])[0])
    return {"Prediction": res}


@app.get('/')
def get_root():
    return {'message': 'Welcome to the Breast Cancer Detection API'}


@app.get('/detect_breast_cancer_path/{data}')
async def detect_breast_cancer(data: str):
    values = []
    for i in data[1:-1].split(","):
        values.append(float(i))
    return prediction(model, values)
