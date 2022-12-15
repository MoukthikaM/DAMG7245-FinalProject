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

def test_model():
    response = client.post(
        "http://127.0.0.1:8080/model", headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer eyJraWQiOiJ6RnFsR1ppaVwva1NmZjNNXC9OVXlkYWFtQjhNb0c5SldkT1cxSE5NdFpqYzQ9IiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiI3ZGExYmUzOS00ODUxLTRiNWMtYTZiMy05ODg2NzM1ZDc2MjIiLCJpc3MiOiJodHRwczpcL1wvY29nbml0by1pZHAudXMtZWFzdC0xLmFtYXpvbmF3cy5jb21cL3VzLWVhc3QtMV9sY05XZzBxeDgiLCJjbGllbnRfaWQiOiI2N242Mmk4cWhycmxnbm50b3FvbjA3OWtldCIsIm9yaWdpbl9qdGkiOiIwN2RiNTM0Ni0wYzFlLTQwOTYtOTUzNC1mMzNhMjJiNzNiMmUiLCJldmVudF9pZCI6IjZmMmFlNTJiLTIzNGMtNDM5ZC1iMmEzLTk4ZGM1YzViMzdhMSIsInRva2VuX3VzZSI6ImFjY2VzcyIsInNjb3BlIjoiYXdzLmNvZ25pdG8uc2lnbmluLnVzZXIuYWRtaW4iLCJhdXRoX3RpbWUiOjE2NzEwNzQ0NTAsImV4cCI6MTY3MTA3ODA1MCwiaWF0IjoxNjcxMDc0NDUwLCJqdGkiOiIxYmUyZGQ4OC1mMWI1LTRmN2UtODc1My0yNzk5ZmM2YmU3ZTQiLCJ1c2VybmFtZSI6IjdkYTFiZTM5LTQ4NTEtNGI1Yy1hNmIzLTk4ODY3MzVkNzYyMiJ9.jYEj92vkSEdW2q4YQofl-tLCQm17zx3a4d0Y8cXACrtnQJU02Us-qGO1A3A7MGEb6rdGPCe8leuFyDGbH9br9q1PBjU6OVohvFzswRL4bbqurK13wK4Cos1dzvnSbZkrHfMT8i1l64oyzPVcnW9OfCMg8VMBpW-zpYKrd1qRQXkTZ54DzXD9tv2XRoHlVqfULukYvfAc2aCuxI3y5Ae2QZFIOwpggO-kfs8QBzduYffmNlrc6moyKZqOJ3RE1iYC7R2GnL_lcysYSo6crn9Qzyj1ufB-e4VeF8Gf-1ZPrjIHyNATj8Ac4BAaxGwU_fqyoLmUHPFdwjUhEOmi3LYYJg',
            'Content-Type': 'application/json'}, json = data
        )
    
    assert response.status_code == 200
    assert response.json() == {'flag': True, 'maxspend': 1254244.0, 'minspend': 613810.0}