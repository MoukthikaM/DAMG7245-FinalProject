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
    minroom, maxroom, minbed, maxbed, minbath, maxbath = customer_df.select(
    floor(min(col("ROOMS"))),
    ceil(max(col("ROOMS"))),
    floor(min(col("BEDROOM2"))),
    ceil(max(col("BEDROOM2"))),
    floor(min(col("BATHROOM"))),
    ceil(max(col("BATHROOM")))
).toPandas().iloc[0, ]

    regions=customer_df.select((col('REGIONNAME'))).toPandas()
# print(suburbs['SUBURB'].unique())
    minroom = int(minroom)
    maxroom = int(maxroom)
    minbed = int(minbed)
    maxbed = int(maxbed)
    minbath = int(minbath)
    maxbath = int(maxbath)

    
    
# Column 1
    with col1:
        st.markdown("#### Search Criteria")
        st.markdown('##')
        region = st.selectbox('REGIONNAME',(regions['REGIONNAME'].unique()))
        suburbs=customer_df.filter(col('REGIONNAME')==region).select((col('SUBURB'))).toPandas()    
        suburb = st.selectbox('SUBURB',(suburbs['SUBURB'].unique()))
        print(suburb)
        rooms = st.slider("ROOMS", minroom, maxroom, (minroom, minroom+5), 1)

        beds = st.slider("BEDROOM", minbed, maxbed, (minbed, minbed+5), 1)

        baths = st.slider("BATH", minbath,
                    maxbath, (minbath, minbath+1), 1)

    # suburb = st.selectbox('SUBURB',(suburbs['SUBURB'].unique()))
    # regions=customer_df.filter(col('SUBURB')==suburb).select((col('REGIONNAME'))).toPandas()
        
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
  "beds": beds,
  "baths": baths,
  "rooms": rooms
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
    
            if minspend==0 or maxspend==0:
                 st.write(f'There is no specific price range for the selected category!!! ')
                
          
            else:
                st.write(f'This house values ranges from ')
                st.metric(label="", value=f"${minspend}")
 
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
