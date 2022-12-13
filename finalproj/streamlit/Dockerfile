FROM python:3.8.10

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit
RUN pip install --upgrade pip wheel
RUN pip install -r requirements.txt
RUN pip install snowflake-snowpark-python[pandas]==0.11.0
COPY . /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "login.py"]

