from airflow.operators.python import PythonVirtualenvOperator
from airflow.decorators import dag, task
from airflow_incremental_tasks import incremental_extract,incremental_load
from airflow_tasks import bulk_train_Linear_regression_task,bulk_train_randomforest_regression_task,bulk_predict_bestmodel,bulk_train_K_nearestneighbours_task
import pendulum
import json
default_args = {
    'owner': 'mouk',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1
}



# session=[]
@dag(
    schedule='0 0 1 * *',
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=['customerspend_incremental_setup_taskflow'])
def houseprice_incremental_setup_taskflow():
    import uuid
    import json
    #Task order - one-time setup
    with open('./include/creds.json') as f:
        connection_parameters = json.load(f)
    state_dict=connection_parameters
    state_dict=incremental_extract(state_dict)
    state_dict=incremental_load(state_dict)
    state_dict1=bulk_train_Linear_regression_task(state_dict)
    state_dict2=bulk_train_randomforest_regression_task(state_dict)
    state_dict_3=bulk_train_K_nearestneighbours_task(state_dict)
    state_dict=bulk_predict_bestmodel(state_dict1,state_dict2,state_dict_3)  
    
    
houseprice_incremental_setup_taskflow()



