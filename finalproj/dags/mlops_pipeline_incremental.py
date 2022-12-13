

def inc_extract(session, download_base_url: str):
   
    import requests
    from zipfile import ZipFile
    from io import BytesIO
    import os
    import pandas as pd
    from datetime import datetime

    file_name_end = '_Melbourne_housing_FULL.csv.zip'
    date_range = pd.period_range(start=datetime.strptime("2016", "%Y"), 
                                 end=datetime.now(), 
                                 freq='Y').strftime("%Y")
    daterange=date_range.to_list()
    length=len(daterange)

    schema_files_to_download_1 = [daterange[length-5]+file_name_end]
    files_to_download = list(schema_files_to_download_1)
    print(schema_files_to_download_1)

    schema_files_to_load = list()
    for file_name in schema_files_to_download_1:
        url = download_base_url+file_name
        print('Downloading and unzipping: '+url)
        r = requests.get(url)
        file = ZipFile(BytesIO(r.content))
        csv_file_name=file.namelist()[0]
        schema_files_to_load.append(csv_file_name)
        file.extract(csv_file_name)
        file.close()

    files_to_load_final = {'schema': schema_files_to_load}
    files_to_load_final

    return files_to_load_final




def inc_load(session,files_to_load_final: dict):
    import pandas as pd
    inc_df = pd.read_csv(files_to_load_final['schema'][0])
    mel_df = session.table('housingprice').toPandas() 
    housing_price=pd.concat([mel_df,inc_df])

    print(f"'housing_price' local dataframe created. Number of records: {len(housing_price)} ")
    snowdf_housing_price = session.createDataFrame(housing_price)
    snowdf_housing_price.write.mode("overwrite").saveAsTable("housingprice") 
    session.table("housingprice").limit(5).show(5)

    # Create a pandas data frame from the Snowflake table
    mel_inc_df = session.table('housingprice').toPandas() 
    print(f"'mel_inc_df' local dataframe created. Number of records: {len(mel_inc_df)} ")
    
    return mel_df