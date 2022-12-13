def database(session, state_dict:dict, prestaged=False):
    
    _ = session.sql('CREATE OR REPLACE DATABASE '+state_dict['database']).collect()
    _ = session.sql('CREATE SCHEMA IF NOT EXISTS '+state_dict['schema']).collect() 
    
    return state_dict
    




def extract_to_stage(session, file_name_end: str, download_base_url: str):
    
    import requests
    from io import BytesIO
    from zipfile import ZipFile
    from datetime import datetime
    import pandas as pd
    
    date_range = pd.period_range(start=datetime.strptime("2016", "%Y"), 
                                 end=datetime.strptime("2017", "%Y"), 
                                 freq='Y').strftime("%Y")
    schema_files_to_download = [date+file_name_end and date+file_name_end for date in date_range.to_list()]
    files_to_download = list(schema_files_to_download)
    print(schema_files_to_download)
 
    schema_files_to_load = list()
    for file_name in schema_files_to_download:
        url = download_base_url+file_name
        print('Downloading and unzipping: '+url)
        r = requests.get(url)
        file = ZipFile(BytesIO(r.content))
        csv_file_name=file.namelist()[0]
        schema_files_to_load.append(csv_file_name)
        file.extract(csv_file_name)
        file.close()

    files_to_load = {'schema': schema_files_to_load}

    return files_to_load


def load(session,files_to_load):
    import pandas as pd
    df_2016 = pd.read_csv(files_to_load['schema'][0])
    df_2017 = pd.read_csv(files_to_load['schema'][1])

    df = pd.concat([df_2016,df_2017])
    df.head()

    snowdf = session.createDataFrame(df)
    snowdf.show()

    snowdf.write.mode("overwrite").saveAsTable("housingprice") 
    session.table("housingprice").show(5)

    # Create a pandas data frame from the Snowflake table
    mel_df = session.table('housingprice').toPandas() 
    print(f"'mel_df' local dataframe created. Number of records: {len(mel_df)} ")

    return mel_df





def train_Linear_regression(session):
    from snowflake.snowpark import functions as F
    from snowflake.snowpark.types import FloatType,IntegerType
    from snowflake.snowpark.version import VERSION
    import pandas as pd 
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import json
    housepricingdf = session.table("HOUSING.PUBLIC.HOUSINGPRICE")
    housepricingdf=housepricingdf.to_pandas()
    housepricingdf.columns = map(lambda x: str(x).upper(), housepricingdf.columns)
    type(housepricingdf)
    cols=['SUBURB','ROOMS','TYPE','METHOD','SELLERG','REGIONNAME','PROPERTYCOUNT','DISTANCE','COUNCILAREA','BEDROOM2','BATHROOM'
           ,'CAR','LANDSIZE','BUILDINGAREA','PRICE']
    housepricingdf=housepricingdf[cols]
    cols_zero = ['PROPERTYCOUNT','DISTANCE','BEDROOM2','BATHROOM','CAR']
    housepricingdf[cols_zero]=housepricingdf[cols_zero].fillna(0)
    housepricingdf.LANDSIZE=housepricingdf.LANDSIZE.fillna(housepricingdf.LANDSIZE.mean())
    housepricingdf.BUILDINGAREA=housepricingdf.BUILDINGAREA.fillna(housepricingdf.BUILDINGAREA.mean())
    housepricingdf.dropna(inplace=True)
    from sklearn.preprocessing import LabelEncoder
    le=[]
    import pickle
    columns=['SUBURB','TYPE','METHOD','SELLERG','REGIONNAME','COUNCILAREA']
    for i in range(0,6):
        le.append(LabelEncoder())
        col=columns[i]
        housepricingdf[col]=le[i].fit_transform(housepricingdf[col])
        filename='le'+str(i)+'.pkl'
        print(filename)
        pickle.dump(le[i], open(filename,'wb'))
    X = housepricingdf.drop(['PRICE'],axis=1)
    y=housepricingdf.PRICE
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    from sklearn.linear_model import LinearRegression
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    from sklearn import metrics
    import numpy as np
    pred = lr.predict(X_test)
    print('MAE:',metrics.mean_absolute_error(y_test,pred))
    print('MSE:',metrics.mean_squared_error(y_test,pred))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,pred)))
    # Calculated R Squared
    print('R^2 =',metrics.explained_variance_score(y_test,pred))
    r2=metrics.explained_variance_score(y_test,pred)
    errors = abs(pred - y_test)
    MSE=round(np.mean(errors),2)
    MAPE=100*(errors/y_test)
    accuracy=round (100 - np.mean(MAPE),2)
    print("Accuracy of linear regression model:")
    print(accuracy)
    with open('lr_pkl', 'wb') as files:
        pickle.dump(lr, files)
    
    
    snowdf_details = session.createDataFrame(housepricingdf)
    snowdf_details.show()
    snowdf_details.write.mode("overwrite").saveAsTable("HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_lr") 

    session.table("HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_lr").show(5)
    return accuracy,r2
    
    

    
def train_randomforest_regression(session):
    from snowflake.snowpark import functions as F
    from snowflake.snowpark.types import FloatType,IntegerType
    from snowflake.snowpark.version import VERSION
    import pandas as pd 
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import json
    housepricingdf = session.table("HOUSING.PUBLIC.HOUSINGPRICE")
    housepricingdf=housepricingdf.to_pandas()
    housepricingdf.columns = map(lambda x: str(x).upper(), housepricingdf.columns)
    type(housepricingdf)
    cols=['SUBURB','ROOMS','TYPE','METHOD','SELLERG','REGIONNAME','PROPERTYCOUNT','DISTANCE','COUNCILAREA','BEDROOM2','BATHROOM'
           ,'CAR','LANDSIZE','BUILDINGAREA','PRICE']
    housepricingdf=housepricingdf[cols]
    cols_zero = ['PROPERTYCOUNT','DISTANCE','BEDROOM2','BATHROOM','CAR']
    housepricingdf[cols_zero]=housepricingdf[cols_zero].fillna(0)
    housepricingdf.LANDSIZE=housepricingdf.LANDSIZE.fillna(housepricingdf.LANDSIZE.mean())
    housepricingdf.BUILDINGAREA=housepricingdf.BUILDINGAREA.fillna(housepricingdf.BUILDINGAREA.mean())
    housepricingdf.dropna(inplace=True)
    from sklearn.preprocessing import LabelEncoder
    le=[]
    import pickle
    columns=['SUBURB','TYPE','METHOD','SELLERG','REGIONNAME','COUNCILAREA']
    for i in range(0,6):
        le.append(LabelEncoder())
        col=columns[i]
        housepricingdf[col]=le[i].fit_transform(housepricingdf[col])
        filename='le'+str(i)+'.pkl'
        print(filename)
        pickle.dump(le[i], open(filename,'wb'))
    X = housepricingdf.drop(['PRICE'],axis=1)
    y=housepricingdf.PRICE
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    from sklearn.ensemble import RandomForestRegressor
    rf_regressor = RandomForestRegressor(bootstrap=True, random_state=0, n_jobs=-1) 
    # fit the regressor with x and y data 
    rf_regressor.fit(X_train,y_train)
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor

    param_grid = dict(n_estimators=[10, 25, 50, 100],
                  max_depth=[5, 10, 20, 30],
                  min_samples_leaf=[1,2,4])
    
    # param_grid = dict(n_estimators=[10],
    #               max_depth=[5],
    #               min_samples_leaf=[1])

    grid = GridSearchCV(rf_regressor, param_grid, cv=10,
                    scoring='neg_mean_squared_error',verbose=2)
    grid.fit(X_train,y_train)
    grid.best_params_
    rf_regressor = RandomForestRegressor(n_estimators=grid.best_params_['n_estimators'],max_depth=grid.best_params_['max_depth'],min_samples_leaf=grid.best_params_['min_samples_leaf'],bootstrap=True, ccp_alpha=0.0,
                                             criterion='mse', n_jobs=2,
                                             oob_score=False, random_state=0,
                                             verbose=1, warm_start=False) 
    rf_regressor.fit(X_train,y_train)
    y_pred = rf_regressor.predict(X_test) 
    from sklearn import metrics
    import numpy as np
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R^2 =',metrics.explained_variance_score(y_test,y_pred))
    r2=metrics.explained_variance_score(y_test,y_pred)
    errors = abs(y_pred - y_test)
    MSE=round(np.mean(errors),2)
    MAPE=100*(errors/y_test)
          
    accuracy=round (100 - np.mean(MAPE),2)
    print("Accuracy of random forest regressor:")
    print(accuracy)
    
    with open('rf_regressor_pkl', 'wb') as files:
        pickle.dump(rf_regressor, files)
        
    snowdf_details = session.createDataFrame(housepricingdf)
    snowdf_details.show()
    snowdf_details.write.mode("overwrite").saveAsTable("HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_rf") 

    session.table("HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_rf").show(5)
    return accuracy,r2

    
    
    
    
def predictbestmodel(session,model,modelname):    
    
    from snowflake.snowpark import functions as F
    from snowflake.snowpark.types import FloatType,IntegerType
    from snowflake.snowpark.version import VERSION
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    from sklearn.neighbors import KNeighborsRegressor

    import pandas as pd 
    import pickle

    def predict_pandas_udf(df: pd.DataFrame) -> pd.Series:
        import pandas as pd
        print(model)
        return pd.Series(model.predict(df)) 
    if modelname=='linear':
        tablename='HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_lr'
        print(tablename)
    elif modelname=='random':
        tablename='HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_rf'
        print(tablename)
    elif modelname=='knn':
        tablename='HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_knn'
        print(tablename)

    housepricingdf = session.table(tablename)
    housepricingdf=housepricingdf.to_pandas()
    X = housepricingdf.drop(['PRICE'],axis=1)
    X=X.columns
    
    from snowflake.snowpark.functions import pandas_udf

    linear_model_vec = pandas_udf(func=predict_pandas_udf,
                                return_type=FloatType(),
                                input_types=[IntegerType(),IntegerType(),IntegerType(),IntegerType(),IntegerType(),IntegerType(),FloatType(),FloatType(),FloatType(),IntegerType(),FloatType(),FloatType(),FloatType(),FloatType()],
                                session=session,
                                packages = ("pandas","scikit-learn"), max_batch_size=200)
    output = session.table(tablename).select(*list(X),
                    linear_model_vec(list(X)).alias('PREDICTED_PRICE'),
                    (F.col('Price')).alias('ACTUAL_PRICE')                                              
                    )

    output.show(5)
    output=output.to_pandas()
    
    
    columns=['SUBURB','TYPE','METHOD','SELLERG','REGIONNAME','COUNCILAREA']
    for i in range(0,6):
        filename='le'+str(i)+'.pkl'
        print(filename)
        le1 = pickle.load(open(filename,'rb'))
        col=columns[i]
        output[col]=le1.inverse_transform(output[col])

    snowdf_details = session.createDataFrame(output)
    snowdf_details.show()
    snowdf_details.write.mode("overwrite").saveAsTable("HOUSING.PUBLIC.HOUSINGPRICE_PREDICTED") 
    
    return snowdf_details




def train_K_nearestneighbours(session):
    from snowflake.snowpark import functions as F
    from snowflake.snowpark.types import FloatType,IntegerType
    from snowflake.snowpark.version import VERSION
    import pandas as pd 
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import json
    housepricingdf = session.table("HOUSING.PUBLIC.HOUSINGPRICE")
    housepricingdf=housepricingdf.to_pandas()
    housepricingdf.columns = map(lambda x: str(x).upper(), housepricingdf.columns)
    type(housepricingdf)
    cols=['SUBURB','ROOMS','TYPE','METHOD','SELLERG','REGIONNAME','PROPERTYCOUNT','DISTANCE','COUNCILAREA','BEDROOM2','BATHROOM'
           ,'CAR','LANDSIZE','BUILDINGAREA','PRICE']
    housepricingdf=housepricingdf[cols]
    cols_zero = ['PROPERTYCOUNT','DISTANCE','BEDROOM2','BATHROOM','CAR']
    housepricingdf[cols_zero]=housepricingdf[cols_zero].fillna(0)
    housepricingdf.LANDSIZE=housepricingdf.LANDSIZE.fillna(housepricingdf.LANDSIZE.mean())
    housepricingdf.BUILDINGAREA=housepricingdf.BUILDINGAREA.fillna(housepricingdf.BUILDINGAREA.mean())
    housepricingdf.dropna(inplace=True)
    from sklearn.preprocessing import LabelEncoder
    le=[]
    import pickle
    columns=['SUBURB','TYPE','METHOD','SELLERG','REGIONNAME','COUNCILAREA']
    for i in range(0,6):
        le.append(LabelEncoder())
        col=columns[i]
        housepricingdf[col]=le[i].fit_transform(housepricingdf[col])
        filename='le'+str(i)+'.pkl'
        print(filename)
        pickle.dump(le[i], open(filename,'wb'))
    X = housepricingdf.drop(['PRICE'],axis=1)
    y=housepricingdf.PRICE
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    
    
    from sklearn.neighbors import KNeighborsRegressor
    knn=KNeighborsRegressor()
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV

    param_grid = {'n_neighbors' : [3,4,5,6,7,10,15] ,    
              'weights' : ['uniform','distance'] ,
              'algorithm' : ['ball_tree', 'kd_tree', 'brute']}
    
    # param_grid = {'n_neighbors' : [3,4]}
    


    grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, refit=True, verbose=2, scoring = 'neg_mean_squared_error')

    grid_knn.fit(X_train, y_train)
    from sklearn import metrics
# Calculated R Squared
    pred_knn = grid_knn.predict(X_test)
 
    
    from sklearn import metrics
    import numpy as np
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_knn))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_knn))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_knn)))
    print('R^2 =',metrics.explained_variance_score(y_test,pred_knn))
    r2=metrics.explained_variance_score(y_test,pred_knn)
    errors = abs(pred_knn - y_test)
    MSE=round(np.mean(errors),2)
    MAPE=100*(errors/y_test)
          
    accuracy=round (100 - np.mean(MAPE),2)
    print("Accuracy of KNN:")
    print(accuracy)
    
    with open('knn_pkl', 'wb') as files:
        pickle.dump(grid_knn, files)
        
    snowdf_details = session.createDataFrame(housepricingdf)
    snowdf_details.show()
    snowdf_details.write.mode("overwrite").saveAsTable("HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_knn") 

    session.table("HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_knn").show(5)
    return accuracy,r2







def train_XGBoost(session):
    from snowflake.snowpark import functions as F
    from snowflake.snowpark.types import FloatType,IntegerType
    from snowflake.snowpark.version import VERSION
    import pandas as pd 
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import json
    housepricingdf = session.table("HOUSING.PUBLIC.HOUSINGPRICE")
    housepricingdf=housepricingdf.to_pandas()
    housepricingdf.columns = map(lambda x: str(x).upper(), housepricingdf.columns)
    type(housepricingdf)
    cols=['SUBURB','ROOMS','TYPE','METHOD','SELLERG','REGIONNAME','PROPERTYCOUNT','DISTANCE','COUNCILAREA','BEDROOM2','BATHROOM'
           ,'CAR','LANDSIZE','BUILDINGAREA','PRICE']
    housepricingdf=housepricingdf[cols]
    cols_zero = ['PROPERTYCOUNT','DISTANCE','BEDROOM2','BATHROOM','CAR']
    housepricingdf[cols_zero]=housepricingdf[cols_zero].fillna(0)
    housepricingdf.LANDSIZE=housepricingdf.LANDSIZE.fillna(housepricingdf.LANDSIZE.mean())
    housepricingdf.BUILDINGAREA=housepricingdf.BUILDINGAREA.fillna(housepricingdf.BUILDINGAREA.mean())
    housepricingdf.dropna(inplace=True)
    from sklearn.preprocessing import LabelEncoder
    le=[]
    import pickle
    columns=['SUBURB','TYPE','METHOD','SELLERG','REGIONNAME','COUNCILAREA']
    for i in range(0,6):
        le.append(LabelEncoder())
        col=columns[i]
        housepricingdf[col]=le[i].fit_transform(housepricingdf[col])
        filename='le'+str(i)+'.pkl'
        print(filename)
        pickle.dump(le[i], open(filename,'wb'))
    X = housepricingdf.drop(['PRICE'],axis=1)
    y=housepricingdf.PRICE
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    
    
    from xgboost import XGBRegressor
    import xgboost as xgb
    regressor=XGBRegressor(eval_metric='rmsle')
    from sklearn.model_selection import GridSearchCV
# set up our search grid
    # param_grid = {"max_depth":    [4, 5],
    #           "n_estimators": [500, 600, 700],
    #           "learning_rate": [0.01, 0.015]}
    param_grid = {"max_depth":    [4],
              "n_estimators": [500],
              "learning_rate": [0.01]}

# try out every combination of the above values
    search = GridSearchCV(regressor, param_grid, cv=5,verbose=50).fit(X_train, y_train)
    print("The best hyperparameters are ",search.best_params_)
    regressor=xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],)

    regressor.fit(X_train, y_train)
    pred_xg = regressor.predict(X_test)
    
    
    
    from sklearn import metrics
    import numpy as np
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_xg))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_xg))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_xg)))
    print('R^2 =',metrics.explained_variance_score(y_test,pred_xg))
    r2=metrics.explained_variance_score(y_test,pred_xg)
    errors = abs(pred_xg - y_test)
    MSE=round(np.mean(errors),2)
    MAPE=100*(errors/y_test)
          
    accuracy=round (100 - np.mean(MAPE),2)
    print("Accuracy of KNN:")
    print(accuracy)
    
    with open('xg_pkl', 'wb') as files:
        pickle.dump(regressor, files)
        
    snowdf_details = session.createDataFrame(housepricingdf)
    snowdf_details.show()
    snowdf_details.write.mode("overwrite").saveAsTable("HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_xg") 

    session.table("HOUSING.PUBLIC.FULL_HOUSINGPRICE_encoded_xg").show(5)
    return accuracy,r2

