# DAMG7245-FinalProject: PREDICTING MELBOURNE HOUSE PRICE STREAMLIT APPLICATION

Codelabs document link : https://codelabs-preview.appspot.com/?file_id=1NjpXEs3k_0-XG7AbUlgu8rw6ISKWl6hPfpkiTSvHy5k#0

## REQUIREMENTS:

1. Create a Snowflake Trial Account

2. Snowpark for Python

3. Docker

4. AWS Account


## SETUP STEPS:

1. Clone the repository and create the environment using the following command :

  		conda env create -f environment.yml
  
 	        conda activate snowpark (from the env file)

2. Launch Jupyter Lab, once jupyter lab is up and running update the creds.json to reflect to your snowflake environment and use the conda environment to run the notebooks.

3. To run streamlit on your terminal, run :

		 streamlit run streamlit.py

  - This is the link for the streamlit hoseted in streamlit cloud. Link for Streamlit: http://ec2-52-207-211-61.compute-1.amazonaws.com:8501/DensityOfHousePrices

4. Run airflow using astro cli :

  - Install Astro CLI using https://docs.astronomer.io/astro/cli/install-cli

  - You can initialise the astro using astro dev init but as the environemnt and folders are already set we can run the below commands to run the airflow:

                astro dev start

                astro dev stop

  - If you make any changes, we can run:

                astro dev restart.

  - Link for Airflow Instance : http://143.198.229.168/home

  - Credentials for airflow : username -admin , password -admin

## CONTRIBUTIONS:

1. Moukthika Manapati :

  - Airflow, Astro

  - Docker

  - Streamlit

  - Model build and train

  - CI/CD

  - Hosting

2. Adhrushta Arashanapalli :

  - Data Preprocessing

  - ELT pipeline

  - Model build and train

  - functions for Bulk load and incremental ingest

  - Auth using AWS Cognito

  - Documentation

=======

