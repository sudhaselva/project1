import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class Details(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

app = FastAPI()

@app.get("/")
async def index():
    return {"message":'This Is Home Page Of API'}
@app.post('/prediction')
async def get_iris_category(data:Details):
    received = data.dict()
    sepal_length = received['sepal_length']
    sepal_width = received['sepal_width']
    petal_length = received['petal_length']
    petal_width = received['petal_width']
    pred_name = model.predict([[sepal_length, sepal_width, petal_length, petal_width]]).tolist()[0]
    return {"prediction":pred_name}

@app.get('/predict')
async def get_cat(sepal_length:float, sepal_width:float, petal_length:float, petal_width:float):
    pred_name = model.predict([[sepal_length, sepal_width, petal_length, petal_width]]).tolist()[0]
    return {"prediction":pred_name}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
