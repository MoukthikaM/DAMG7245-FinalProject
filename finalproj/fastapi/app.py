from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import *
import numpy as np
from fastapi import FastAPI,Form,Request,Header,Depends
import json
from pydantic import BaseModel
from auth.decodetoken import lambda_handler
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
security = HTTPBearer()
app = FastAPI()

class SqlQuery(BaseModel):
    suburb: str
    asl: tuple
    toa: tuple
    lom: tuple
    tow: tuple
    car: tuple
    ba: tuple

    
    
@app.post("/model")
async def model(query: SqlQuery,credentials: HTTPAuthorizationCredentials = Security(security)):
  token = credentials.credentials
  event = {}
  event['token']=token
  print(event)
  flag,claims=lambda_handler(event,None)  
  if flag==True:
    with open("./creds.json") as f:
        connection_parameters = json.load(f)
    session = Session.builder.configs(connection_parameters).create()
    customer_df = session.table('HOUSING.PUBLIC.HOUSINGPRICE_PREDICTED')
    print(query)
    suburb=query.suburb
    lom=query.lom
    asl=query.asl
    toa=query.toa
    tow=query.tow
    ba=query.ba
    car=query.car
    minspend, maxspend = customer_df.filter(
         (col("SUBURB") == suburb) &
          ( col("ROOMS") <= asl[1]) & (  col("ROOMS") > asl[0])
        # & (col("LANDSIZE") <= toa[1]) & (col("LANDSIZE") > toa[0])
        & (col("BEDROOM2") <= tow[1]) & (col("BEDROOM2") > tow[0])
        & (col("BATHROOM") <= lom[1]) & (col("BATHROOM") > lom[0])
        & (col("CAR") <= car[1]) & (col("CAR") > car[0])
        & (col("BUILDINGAREA") <= ba[1]) & (col("BUILDINGAREA") > ba[0])
    ).select(trunc(min(col('PREDICTED_PRICE'))), trunc(max(col('PREDICTED_PRICE')))).toPandas().iloc[0, ]
    
    print(minspend,maxspend)
    return {"minspend":minspend,"maxspend":maxspend,"flag":True}
  elif flag==False:
        print(claims['username'])
        uuid=claims['username']
        return {"username":uuid,"flag":False}



# @app.post("/authtest")
# async def model(credentials: HTTPAuthorizationCredentials = Security(security)):
#     token = credentials.credentials
#     event = {}
#     event['token']=token
#     print(event)
#     if(lambda_handler(event,None)):
#          return "AUTH"
#     # if(lambda_handler(event,None)):
#     #     return "Authenticated"


