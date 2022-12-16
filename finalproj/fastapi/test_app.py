from fastapi.testclient import TestClient
from app import app
import json
import requests

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

def test_model():

    url = "https://83vrb97fv1.execute-api.us-east-1.amazonaws.com/beta/signIn"

    payload = json.dumps({
  "email": "adhrushta@gmail.com",
  "password": "RandomPass1@"    
    })
    headers = {
  'Accept': 'application/json',
  'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    dict=json.loads(response.text)
    accesstoken = dict['body']['data']['AuthenticationResult']['AccessToken']
    # print(response.text)
    Auth='Bearer '+accesstoken
    print(Auth)
    response = client.post(
        "http://127.0.0.1:8080/model", headers = {
            'Accept': 'application/json',
            'Authorization': Auth,
            'Content-Type': 'application/json'}, json = data
        )
    
    assert response.status_code == 200
    # assert response.json() == {'flag': True, 'maxspend': 1302940.0, 'minspend': 613746.0}