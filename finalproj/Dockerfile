FROM quay.io/astronomer/astro-runtime:7.0.0
USER root
RUN apt-get -y update \
  && apt-get -y upgrade \
  && apt-get install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev wget -y \
  && wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz \
  && tar -xf Python-3.8.12.tar.xz \
  && mv Python-3.8.12 /opt/Python3.8.12
WORKDIR /opt/Python3.8.12/
RUN ./configure \
#--enable-optimizations --enable-shared
  && make \
  && make altinstall \
  && ldconfig /opt/Python3.8.12 \
  && pip3.8 install snowflake-snowpark-python[pandas]==0.11.0 jupyterlab
USER astro
WORKDIR /usr/local/airflow
RUN /usr/local/bin/python3.8 -m pip install --upgrade pip
RUN /usr/local/bin/python3.8 -m pip install -r requirements.txt
