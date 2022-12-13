
import json
import time
import urllib.request
from jose import jwk, jwt
from jose.utils import base64url_decode

region = 'us-east-1'
userpool_id = 'us-east-1_lcNWg0qx8'
app_client_id = '67n62i8qhrrlgnntoqon079ket'
keys_url = 'https://cognito-idp.{}.amazonaws.com/{}/.well-known/jwks.json'.format(region, userpool_id)
# instead of re-downloading the public keys every time
# we download them only on cold start
# https://aws.amazon.com/blogs/compute/container-reuse-in-lambda/
with urllib.request.urlopen(keys_url) as f:
  response = f.read()
keys = json.loads(response.decode('utf-8'))['keys']

def lambda_handler(event, context):
    token = event['token']
    # get the kid from the headers prior to verification
    headers = jwt.get_unverified_headers(token)
    kid = headers['kid']
    # search for the kid in the downloaded public keys
    key_index = -1
    for i in range(len(keys)):
        if kid == keys[i]['kid']:
            key_index = i
            break
    if key_index == -1:
        print('Public key not found in jwks.json')
        return False
    # construct the public key
    public_key = jwk.construct(keys[key_index])
    # get the last two sections of the token,
    # message and signature (encoded in base64)
    message, encoded_signature = str(token).rsplit('.', 1)
    # decode the signature
    decoded_signature = base64url_decode(encoded_signature.encode('utf-8'))
    # verify the signature
    if not public_key.verify(message.encode("utf8"), decoded_signature):
        print('Signature verification failed')
        return False
    
    # since we passed the verification, we can now safely
    # use the unverified claims
    claims = jwt.get_unverified_claims(token)
    # additionally we can verify the token expiration
    if time.time() > claims['exp']:
        print('Token is expired')
        return False,claims
    # and the Audience  (use claims['client_id'] if verifying an access token)
    if claims['client_id'] != app_client_id:
        print('Token was not issued for this audience')
        return False,claims
    print('Signature successfully verified')
    # now we can use the claims
    print(claims)
    return True,claims
        

# event = {'token': 'eyJraWQiOiJ6RnFsR1ppaVwva1NmZjNNXC9OVXlkYWFtQjhNb0c5SldkT1cxSE5NdFpqYzQ9IiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiIzNDIxYmVhMC1iMmMzLTRkN2EtODE2YS02YjdhZDA1ZGUwNDgiLCJpc3MiOiJodHRwczpcL1wvY29nbml0by1pZHAudXMtZWFzdC0xLmFtYXpvbmF3cy5jb21cL3VzLWVhc3QtMV9sY05XZzBxeDgiLCJjbGllbnRfaWQiOiI2N242Mmk4cWhycmxnbm50b3FvbjA3OWtldCIsIm9yaWdpbl9qdGkiOiJmNGM3NjJhOC0wZjE0LTRkNDctOGM0Mi1mMGEyZmFhZjM1YzAiLCJldmVudF9pZCI6ImZjNTY2OWUxLWZhZjMtNDQyMC1hZDFiLTJmOTJlNDQ4MGZjZiIsInRva2VuX3VzZSI6ImFjY2VzcyIsInNjb3BlIjoiYXdzLmNvZ25pdG8uc2lnbmluLnVzZXIuYWRtaW4iLCJhdXRoX3RpbWUiOjE2NzA3Mjc2MDMsImV4cCI6MTY3MDczMTIwMywiaWF0IjoxNjcwNzI3NjAzLCJqdGkiOiI4M2JiOTk3Ny05MDM4LTQ5YzAtOTU0OS02NzM3ZTlkNzM3OWIiLCJ1c2VybmFtZSI6IjM0MjFiZWEwLWIyYzMtNGQ3YS04MTZhLTZiN2FkMDVkZTA0OCJ9.HKzT8wFVySJiyttE89mCFu-DuKpEdXRM7sKcVQ00lpshI0W4Eccy6OajEopHP32U1ROE-qAFhcKHMwxvE1felkMmxmggH56DFc1Iaw-Wv2LooLlXwLIzCwXFRz_MqlC72tG8tGIr7CA7VlFUb8dvdACTp2laiTSjPb-FqiVsSzUjxvJpC19_jWirVhmMFPva5mko1acikIpNplvXrHOvLJgBudAFa5wnZ4RBTOWuX4syDlPhnYbi59DVWSOeg5a_ajRjOskvtP1ZTQRx9_Gc4_3hNqyC5U7mNBazqS30JB76vhGhQmSWo7dzvK4j_GHbIc8Z4s5fc4U_0tGLFaXq-A'}
# lambda_handler(event, None)
