import streamlit as st  
import json
import requests
st.set_page_config(
      page_title="SEVIR App",
      page_icon="👋",
)

headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logOutSection = st.container()
# st.session_state['user']=''



def loginuser(username,password):
        url = "https://83vrb97fv1.execute-api.us-east-1.amazonaws.com/beta/signIn"
        print(password)
        payload = json.dumps({
  "email": username,
  "password": password
})
        headers = {
  'Content-Type': 'application/json'
}
        print(username)
        response = requests.request("POST", url, headers=headers, data=payload)
        dict=json.loads(response.text)
        print(dict)
        # print(response.text)
        if dict['statusCode'] == 200: 
             st.session_state['auth_code']=dict['body']['data']['AuthenticationResult']['AccessToken']
             st.session_state['refreshtoken']=dict['body']['data']['AuthenticationResult']['RefreshToken']   
             return response.text
        else:
                if dict['body']['message']=='Sorry! The credentials does not match.':
                    return False
                # elif dict['detail']=='Invalid token':
                #     st.session_state['auth_code']=new_token
                # elif dict['detail']=='Invalid username':
                #     return False



def refreshtoken():

        url = "http://fastapi:8000/refresh_token"

        payload={}
        headers = {
  'Authorization': 'Bearer '+ st.session_state['auth_code']
}

        response = requests.request("GET", url, headers=headers, data=payload)
        dict=json.loads(response.text)
        # print(response.text)
        return dict['token']


def show_main_page():
    with mainSection:
      st.title("Welcome {} 🌟".format(st.session_state['user']))
      print(st.session_state['user'])
      st.sidebar.success("Select a page above.")
#       st.write("Welcome user",st.session_state['user'])
 
        
          



def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False
    st.session_state['auth_code'] = ""
    st.session_state['user'] = ""
    st.session_state['refreshtoken']=""
    
def show_logout_page():
    loginSection.empty();
    with logOutSection:
        st.button ("Log Out", key="logout", on_click=LoggedOut_Clicked)
        

def LoggedIn_Clicked(userName, passWord):
    if loginuser(userName, passWord):
        st.session_state['loggedIn'] = True
        st.session_state['user'] = userName
        
    else:
        st.session_state['loggedIn'] = False;
        st.error("Invalid user name or password")

def show_login_page():
    with loginSection:
        if st.session_state['loggedIn'] == False:
            userName = st.text_input (label="username", key='ulogin',placeholder="Enter your user name")
            passWord = st.text_input (label="password", key='login',placeholder="Enter password", type="password")
            if userName and passWord:
                st.button ("Login", on_click=LoggedIn_Clicked, args= (userName, passWord))



with headerSection:
    st.title("SEVIR Cloud Application")
    #first run will have nothing in session_state
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False
        show_login_page() 
    else:
        if st.session_state['loggedIn']:
            show_logout_page()   
            show_main_page()

        else:
            show_login_page()



    
    

