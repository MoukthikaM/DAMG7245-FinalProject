FROM tiangolo/uvicorn-gunicorn:python3.8

RUN mkdir /fastapi
COPY requirements.txt /fastapi
WORKDIR /fastapi

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install snowflake-snowpark-python[pandas]==0.11.0
COPY . /fastapi
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

