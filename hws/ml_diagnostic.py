#ML Diagnostic Python

import pandas as pd
import urllib
import re
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from pandas.io.json import json_normalize
import json
import requests


pd.set_option("display.max_columns",34)

raw_311_sub = pd.read_csv('chicago_combined.csv')

sort_sub_311 = raw_311_sub.sort('creation_date')

sort_sub_311 = sort_sub_311.drop('Unnamed: 0', axis=1)

def soda_loop(url):
    offset_num = 0
    df = pd.DataFrame()
    count = 0
    use_url = url
    while len(pd.read_json(use_url)):

        df = pd.concat([df,pd.read_json(use_url)])
        
        use_url = url
  
        offset_num+=50000
   
        use_url=url+'&$offset={}'.format(offset_num)
                
        count+=1
        
        print(use_url)
        
    return df
'''
#query 311 graffiti data
raw_graffiti = soda_loop('https://data.cityofchicago.org/resource/qr2f-jf4g.json?$$app_token=JTo1u34jDTjIhmflT0bRfaVb0&$limit=50000')

#query 311 vacant lot and abandon buildings reported
raw_vac = soda_loop('https://data.cityofchicago.org/resource/4mhb-6abn.json?$$app_token=JTo1u34jDTjIhmflT0bRfaVb0&$limit=50000')


#query 311 potholes reported
raw_ph = soda_loop('https://data.cityofchicago.org/resource/ej8m-yavt.json?$$app_token=JTo1u34jDTjIhmflT0bRfaVb0&$limit=50000')

#query 311 sanitation code complaints
raw_san = soda_loop('https://data.cityofchicago.org/resource/tybt-vtit.json?$$app_token=JTo1u34jDTjIhmflT0bRfaVb0&$limit=50000')

raw_vac['street_address'] = raw_vac[raw_vac.columns[[0,1,2,3]]].apply(lambda x: ' '.join(x.astype(str).astype(str)),axis=1)

raw_vac = raw_vac.drop(raw_vac.columns[[0,1,2,3]],1)

raw_vac_e = raw_vac.rename(index=str, columns={'date_service_request_was_received':'creation_date','service_request_type':'type_of_service_request'})
#raw_vac_e = raw_vac.rename(index=str, columns={'service_request_type':'type_of_service_request'})
dfs = [raw_graffiti, raw_vac_e, raw_ph, raw_san]
raw_311 = pd.concat(dfs)
raw_311_sub = raw_311[['creation_date','completion_date','community_area','street_address','type_of_service_request', 'latitude', 'longitude','police_district','ward','is_building_open_or_boarded_','number_of_potholes_filled_on_block','what_type_of_surface_is_the_graffiti_on_','what_is_the_nature_of_this_code_violation_','x_coordinate','y_coordinate','zip']].copy()

#311 call by type of service request
pd.DataFrame(raw_311['type_of_service_request'].value_counts())

#Plot of 311 Calls by type of service request
raw_311_sub['type_of_service_request'].value_counts().plot("bar")
plt.figure()
plt.show()

#Plot of 311 Calls by Community Area
raw_311_sub['community_area'].hist()
plt.figure()
plt.show()

#DataFrame of area calls by type of call
by_area = raw_311_sub.groupby(['community_area', 'type_of_service_request']).size().to_frame()

#Plot of graffiti calls by type of surface
raw_graffiti['what_type_of_surface_is_the_graffiti_on_'].value_counts().plot('bar')
plt.figure()
plt.show()

#Table of potholes based on number of potholes filled on block
raw_ph.groupby('number_of_potholes_filled_on_block').size().to_frame()

#Plot of potholes based on number of potholes filled on block
raw_ph['number_of_potholes_filled_on_block'].value_counts().plot('bar')
plt.figure()
plt.show()

raw_vac['is_building_open_or_boarded_'].value_counts().plot('bar')
plt.figure()
plt.show()
'''

def find_fips(df):
    
    acs_df = pd.DataFrame()

    for row in df.itertuples():
        
        
        print(row[5])
        if row[5] == 'Graffiti Removal':
            continue
        
        
        print (row[1][6])
        
        if int(row[1][6]) <= 6:
            lat = row[6]
            long = row[7]

            if str(lat) == 'nan':
                continue
                
            fcc_api_call = 'http://data.fcc.gov/api/block/find?format=json&latitude={}&longitude={}&showall=true'.format(lat,long)
            

            print(row[0])
            
            
            
            #B02008_001E: White Alone or in Combination with other races 
            #B02009_001E: Black Or African American Alone Or In Combination With One Or More Other Races 
            #B02011_001E:  Asian Alone Or In Combination With One Or More Other Races
            
            #acs_call = 'http://api.census.gov/data/2015/acs5?get=NAME,B02008_001E,B02009_001E,B02011_001E&for=block+group:{}&in=state:17&in=county:031&in=tract:{}&key=d53148b74dc29b6d066fe5ecbf2f030e63183f04'.format(block_group,tract)
            
            #acs_return = pd.DataFrame(pd.read_json(acs_call).ix[1]).transpose().reset_index()
            
            #street_type_acs = acs_return.join(pd.DataFrame(df.ix[r_index]).transpose().reset_index(), lsuffix='test')
            try:
                fcac = pd.read_json(fcc_api_call)
                fips = str(int(fcac['Block']['FIPS']))
            except ValueError:
                d = json.loads(requests.get(fcc_api_call).text)
                fips = pd.DataFrame(d['Block']['intersection'])['FIPS'][0]
            
            #block group digits
            block_group = fips[-4]
            
            #tract digits
            tract = fips[-10:-4]
            
            if fips == 'nan':
                continue
            #fcac = pd.read_json(fcc_api_call)
            
            df_addtract = pd.DataFrame(df.ix[row[0]]).transpose().reset_index().join(pd.DataFrame({'tract':tract, 'block group':block_group}, index=[0]))
            
            acs_df = pd.concat([acs_df, df_addtract])
                                         
            #if r_index == 97:
            #    return acs_df
        else:
            break
        
        
    
    return acs_df

better_result = find_fips(sort_sub_311)
