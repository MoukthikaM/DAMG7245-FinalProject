from airflow.decorators import task

@task.virtualenv(python_version=3.8)
def incremental_extract(state_dict:dict,  
             prestaged=False):
    from dags.mlops_pipeline_incremental import inc_extract
    import requests
    from io import BytesIO
    from zipfile import ZipFile
    
    from snowflake.snowpark.session import Session
    import json
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)  
    session = Session.builder.configs(connection_parameters).create()
    download_base_url = "https://s3.amazonaws.com/damgbucket/project/"
    files_to_load=inc_extract(session=session, 
                download_base_url=download_base_url)
    state_dict['files_to_load']=files_to_load
    return state_dict 

  
@task.virtualenv(python_version=3.8)
def incremental_load(state_dict:dict,prestaged=False):
    import pandas as pd
    from dags.mlops_pipeline_incremental import inc_load
    from snowflake.snowpark.session import Session
    import json
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)  
    session = Session.builder.configs(connection_parameters).create()
    files_to_load=state_dict['files_to_load']
    print(files_to_load)
    stage_table_names = inc_load(session,files_to_load)
    
    return state_dict

