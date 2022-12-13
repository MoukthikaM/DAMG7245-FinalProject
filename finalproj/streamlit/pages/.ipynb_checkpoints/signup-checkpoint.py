import streamlit as st  
import json
import requests

st.session_state["auth_code"]=""
st.session_state['loggedIn'] = False
def signup_user(name,username,password):
       
        url = "https://83vrb97fv1.execute-api.us-east-1.amazonaws.com/beta/signup"
        payload = json.dumps({
  "name": name,
  "email": username,
  "password": password
})
        headers = {
  'Content-Type': 'application/json'
}

        response = requests.request("POST", url, headers=headers, data=payload)
        # print(response.text)
        dict=json.loads(response.text)
        # print(dict)
        return dict

name = st.text_input("Name")
username = st.text_input("User Name")
password = st.text_input("Password",type='password')

if st.button('SignUp'):    
    if username and password and name:        
        dict=signup_user(name,username,password)
        print(dict)
        if dict['statusCode']==200:
            st.success("You have successfully created an account.Go to the Login Menu to login")
        elif dict['statusCode']==500: 
            st.error("Enter Valid Email")
        elif dict['statusCode']==422: 
            st.error("Signup Failed username already present")
    else:
         st.error("Please Fill the Details")   
        
