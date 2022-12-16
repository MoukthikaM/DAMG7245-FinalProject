import streamlit as st
import pandas as pd
import numpy as np
import json

from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import *


with open('creds.json') as f:
    connection_parameters = json.load(f)    
session = Session.builder.configs(connection_parameters).create()

session.sql("SELECT count(*) FROM HOUSING.PUBLIC.HOUSINGPRICE").collect()
housepricingdf = session.table("HOUSING.PUBLIC.HOUSINGPRICE")
housepricingdf=housepricingdf.to_pandas()

st.subheader('Melbourne Heatmap of the House Prices')

# Creating a new dataset with the Latitude and Longitude
latitude = housepricingdf['Lattitude']
longitude = housepricingdf['Longtitude']


import plotly.express as px
fig = px.density_mapbox(housepricingdf, lat=latitude, lon=longitude, z=housepricingdf['Price'],
                        center=dict(lat=-37.823002, lon=144.998001), zoom=9,
                        mapbox_style="stamen-terrain",
                        radius=20,
                        opacity=0.5)
fig.update_layout(title_text='Melbourne Heatmap of the House Prices', title_x=0.5, title_font=dict(size=32))
fig.show(renderer="iframe_connected")
st.plotly_chart(fig)
