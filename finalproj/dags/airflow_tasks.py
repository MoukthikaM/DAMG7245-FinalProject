from airflow.decorators import task


@task.virtualenv(python_version=3.8)
def reset_database(state_dict:dict, prestaged=False)->dict:
    from snowflake.snowpark.session import Session
    import json
    from dags.mlops_pipeline import database
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)  
    session = Session.builder.configs(connection_parameters).create()
    database(session, state_dict)
    return state_dict




@task.virtualenv(python_version=3.8)
def bulk_extract(state_dict:dict,  
             prestaged=False):
    from dags.mlops_pipeline import extract_to_stage
    import requests
    from io import BytesIO
    from zipfile import ZipFile
    from snowflake.snowpark.session import Session
    import json
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)  
    session = Session.builder.configs(connection_parameters).create()
    file_name_end = '_Melbourne_housing_FULL.csv.zip'
    download_base_url = "https://s3.amazonaws.com/damgbucket/project/"
    files_to_load=extract_to_stage(session=session, 
                file_name_end=file_name_end, 
                download_base_url=download_base_url)
    state_dict['files_to_load']=files_to_load
    return state_dict 

  
@task.virtualenv(python_version=3.8)
def bulk_load(state_dict:dict,prestaged=False):
    import pandas as pd
    from dags.mlops_pipeline import load
    from snowflake.snowpark.session import Session
    import json
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)  
    session = Session.builder.configs(connection_parameters).create()
    files_to_load=state_dict['files_to_load']
    stage_table_names = load(session,files_to_load)
    
    return state_dict




@task.virtualenv(python_version=3.8)
def bulk_train_Linear_regression_task(state_dict:dict)-> dict: 
    print("BYE from train predict")
    
    from snowflake.snowpark.session import Session
    from dags.mlops_pipeline import train_Linear_regression
    import json
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)
    session = Session.builder.configs(connection_parameters).create()  
    accuracy,r2=train_Linear_regression(session)
    print("REACHED")
    state_dict['accuracy']=accuracy
    state_dict['r2']=r2
    # state_dict['x']=X
    session.close()
    return state_dict



@task.virtualenv(python_version=3.8)
def bulk_train_randomforest_regression_task(state_dict:dict)-> dict: 
    print("BYE from train predict")
    
    from snowflake.snowpark.session import Session
    from dags.mlops_pipeline import train_randomforest_regression
    import json
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)
    session = Session.builder.configs(connection_parameters).create()  
    accuracy,r2=train_randomforest_regression(session)
    print("REACHED")
    state_dict['accuracy']=accuracy
    state_dict['r2']=r2
    # state_dict['x']=X
    session.close()
    return state_dict





@task.virtualenv(python_version=3.8)
def bulk_train_K_nearestneighbours_task(state_dict:dict)-> dict: 
    print("BYE from train predict")
    
    from snowflake.snowpark.session import Session
    from dags.mlops_pipeline import train_K_nearestneighbours
    import json
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)
    session = Session.builder.configs(connection_parameters).create()  
    accuracy,r2=train_K_nearestneighbours(session)
    print("REACHED")
    state_dict['accuracy']=accuracy
    state_dict['r2']=r2
    # state_dict['x']=X
    session.close()
    return state_dict




@task.virtualenv(python_version=3.8)
def bulk_train_XGBoost_task(state_dict:dict)-> dict: 
    print("BYE from train predict")
    
    from snowflake.snowpark.session import Session
    from dags.mlops_pipeline import train_XGBoost
    import json
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)
    session = Session.builder.configs(connection_parameters).create()  
    accuracy,r2=train_XGBoost(session)
    print("REACHED")
    state_dict['accuracy']=accuracy
    state_dict['r2']=r2
    session.close()
    return state_dict




@task.virtualenv(python_version=3.8)
def bulk_predict_bestmodel(state_dict1:dict,state_dict2:dict,state_dict3:dict)-> dict: 
    print(state_dict1)
    print(state_dict2)
    print(state_dict3)

    lr_r2=state_dict1['r2']
    rf_r2=state_dict2['r2']
    knn_r2=state_dict3['r2']
    from snowflake.snowpark.session import Session
    from dags.mlops_pipeline import predictbestmodel
    import json
    import pickle
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)
    session = Session.builder.configs(connection_parameters).create()
    if lr_r2 > rf_r2 and lr_2 > knn_r2  :
        print("linear regression is considered as the best model")
        
        with open('lr_pkl' , 'rb') as f:
            lr = pickle.load(f)
        predictbestmodel(session,lr,'linear')
        
        return state_dict1
    elif lr_r2 < rf_r2 and rf_r2 > knn_r2 :
        print("Random Forest Regression Model is considered as the best model")
        
        with open('rf_regressor_pkl' , 'rb') as f:
            rf = pickle.load(f)
            predictbestmodel(session,rf,'random')
        return state_dict2
    elif knn_r2>lr_r2 and knn_r2 > rf_r2 :
        print("KNN is considered as the best model")
        
        with open('knn_pkl' , 'rb') as f:
            knn = pickle.load(f)
            predictbestmodel(session,knn,'knn')
        return state_dict3
