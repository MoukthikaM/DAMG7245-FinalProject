from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)
print(client)

data = {
        "suburb" :"Moorabbin",
        "asl":[1,6],
        "toa":[0, 5],
        "lom":[0, 1], 
        "tow":[0, 5],
        "car":[0.0, 4.0],
        "ba":[0.0, 500.0]
            }

def test_signIn():
    response=   

def test_model():
    response = client.post(
        "/model",
         json = data
        )
    # assert response.status_code == 200
    assert response.json() == {
                "minspend": 613810.0,
                "maxspend": 1254244.0,
                "flag": 'true'
                }