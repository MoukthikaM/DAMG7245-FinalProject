import json
import boto3
import botocore.exceptions
import os
import hmac
import hashlib
import base64
    
def get_secret_hash(username):
    msg = username + os.environ['clientId']
    dig = hmac.new(str(os.environ['clientSecret']).encode('utf-8'), 
        msg = str(msg).encode('utf-8'), digestmod=hashlib.sha256).digest()
    d2 = base64.b64encode(dig).decode()
    return d2
    
def returnResponse(code, message, data = None):
    return {
            'statusCode': code,
            'body': {
                'message': message,
                'data': data
            }
        }

def lambda_handler(event, context):
    
    client = boto3.client('cognito-idp')
    
    try:
        response = client.initiate_auth(
            AuthFlow='REFRESH_TOKEN_AUTH',
            AuthParameters={
                'REFRESH_TOKEN': event['refresh_token'],
                'SECRET_HASH': get_secret_hash(event['username']) #when you have an “@” in the username you get that error on the REFRESH_TOKEN_AUTH call. Cognito generates a UUID-style username for them. And you have to use that during the refresh call.
            },
            ClientId=os.environ['clientId']
        )
        
        print(response)
        
        return returnResponse(200, 'Token Refreshed successfully.', response)
        
    except client.exceptions.NotAuthorizedException as e:
        print(str(e))
        return returnResponse(422, 'Sorry! The refresh token credetials does not match.')
        
    except Exception as e:
        print(str(e))
        return returnResponse(500, "Whoops! Something went wrong. Please try again later")

