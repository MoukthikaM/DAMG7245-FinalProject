import streamlit as st
import json
import requests
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import *

st.title("Model As a Service")


def model_clicked():


# %%
# Create a session to Snowflake with credentials
    with open("./creds.json") as f:
        connection_parameters = json.load(f)
    session = Session.builder.configs(connection_parameters).create()


# %%
# Header
    head1, head2 = st.columns([8, 1])

    with head1:
        st.header("House Price Prediction Model in Melbourne")
    with head2:
        st.markdown(
        f' <img src="https://api.nuget.org/v3-flatcontainer/snowflake.data/0.1.0/icon" width="50" height="50"> ',
        unsafe_allow_html=True)

    st.markdown('##')
    st.markdown('##')
    # %%
# Customer Spend Slider Column
    col1, col2, col3 = st.columns([4, 1, 10])

    customer_df = session.table('HOUSING.PUBLIC.HOUSINGPRICE_PREDICTED')

# Read Data
    minasl, maxasl, mintoa, maxtoa, mintow, maxtow, minlom, maxlom,mincar,maxcar,minba,maxba = customer_df.select(
    floor(min(col("ROOMS"))),
    ceil(max(col("ROOMS"))),
    floor(min(col("DISTANCE"))),
    ceil(max(col("DISTANCE"))),
    floor(min(col("BEDROOM2"))),
    ceil(max(col("BEDROOM2"))),
    floor(min(col("BATHROOM"))),
    ceil(max(col("BATHROOM"))),
    floor(min(col("CAR"))),
    ceil(max(col("CAR"))),
    floor(min(col("BUILDINGAREA"))),
    ceil(max(col("BUILDINGAREA")))
).toPandas().iloc[0, ]

    regions=customer_df.select((col('REGIONNAME'))).toPandas()
# print(suburbs['SUBURB'].unique())
    minasl = int(minasl)
    maxasl = int(maxasl)
    mintoa = int(mintoa)
    maxtoa = int(maxtoa)
    mintow = int(mintow)
    maxtow = int(maxtow)
    minlom = int(minlom)
    maxlom = int(maxlom)
    
    
# Column 1
    with col1:
        st.markdown("#### Search Criteria")
        st.markdown('##')
        asl = st.slider("ROOMS", minasl, maxasl, (minasl, minasl+5), 1)
    #st.write("Session Length ", asl)
        toa = st.slider("LANDSIZE", mintoa, maxtoa, (mintoa, mintoa+5), 1)
    #st.write("Time on App ", toa)
        tow = st.slider("BEDROOM", mintow, maxtow, (mintow, mintow+5), 1)
    #st.write("Time on Website ", tow)
        lom = st.slider("BATH", minlom,
                    maxlom, (minlom, minlom+1), 1)
        car = st.slider("CAR", mincar,
                    maxcar, (mincar, mincar+4), 1.0)
        ba = st.slider("BUILDINGAREA", minba,
                    maxba, (minba, minba+500), 1.0)
    # suburb = st.selectbox('SUBURB',(suburbs['SUBURB'].unique()))
    # regions=customer_df.filter(col('SUBURB')==suburb).select((col('REGIONNAME'))).toPandas()
        region = st.selectbox('REGIONNAME',(regions['REGIONNAME'].unique()))
        suburbs=customer_df.filter(col('REGIONNAME')==region).select((col('SUBURB'))).toPandas()    
        suburb = st.selectbox('SUBURB',(suburbs['SUBURB'].unique()))
        print(suburb)
    #st.write("Length of Membership ", lom)

# Column 2 (3)
    with col3:

        st.markdown("#### HOUSE PRICE PREDICTION")
        st.markdown('##')
    
        event = {}
        event['token']=st.session_state['auth_code']
        url = "http://fastapi:8000/model"

        payload = json.dumps({
  "suburb": suburb,
  "asl": asl,
  "toa" : toa,
  "tow": tow,
  "lom": lom,
   "car": car,
    "ba": ba
})
    # payload = jsonreq
        print(payload)
        headers = {
  'Authorization': 'Bearer '+ st.session_state['auth_code']
}

        price = requests.request("POST", url, headers=headers, data=payload)
        print(price)
        price=price.json()
        print(st.session_state)
        if price["flag"]==True:
           

            minspend=price["minspend"]
            maxspend=price["maxspend"]
    
    
    

            st.write(f'This house values ranges from ')
            st.metric(label="", value=f"${minspend}")
    #st.write("and")
            st.metric(label="and", value=f"${maxspend}")

            st.markdown("---")
        elif price["flag"]==False:
            print(st.session_state)
            uuid=price
            print(uuid)
            url = "https://83vrb97fv1.execute-api.us-east-1.amazonaws.com/beta/refreshtoken"

            payload = json.dumps({
  "refresh_token": st.session_state['refreshtoken'],
  "username": uuid
})
            headers = {
  'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)
            dict=json.loads(response.text)
            print(dict)
       
            if dict['statusCode'] == 200: 
                 st.session_state['auth_code']=dict['body']['data']['AuthenticationResult']['AccessToken']  
                 model_clicked()








logOutSection=st.container()
def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False
    st.session_state['auth_code'] = ""
    st.session_state['user'] = ""
    
def show_logout_page():
    with logOutSection:
        st.button ("Log Out", key="logout", on_click=LoggedOut_Clicked)


if 'auth_code' not in st.session_state:
    st.write("Please Login")

elif st.session_state["auth_code"]:
     model_clicked()
     show_logout_page()
else:
     st.write("Please Login")
