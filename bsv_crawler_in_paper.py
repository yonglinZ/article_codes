"""
2018/12/25

@author: ZYL_rcees

BSVP_cpt v2.0
- Baidu street veiw panorama crawler and processing tool
- Version 2.0

"""
import os
import time
import json
import math
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pylab as plt
from pylab import *
from urllib.request import urlopen
from urllib.parse import quote, unquote
import multiprocessing
from multiprocessing import Pool
import progressbar

data_path = r'E:/project_y/data'
folder_path = r'E:/project_y/Beijing/Beijign_BSVs_merge/'# output folder
city = 'BJ_' # Beijing

data = pd.read_table(os.path.join(data_path,'GZ_road100_points.txt'),delimiter=',')
s1 = data['FID']
s2 = data['lat']
s3 = data['lon']
data = pd.concat([s1,s2,s3],axis=1)
data.columns = ['FID','lat','lon']

key = '8ljimGEfgienvbpHSDFOBag5N3K49' 
key30 = '6IMDbRWqr6Tu8U7PdqwkRGPcXKvVP' 


def div_df4(df,start): 
       start = start
       index=math.ceil(len(df)/4)
       a=df.ix[start:start+index-1,:]
       b=df.ix[start+index:start+index*2-1,:]
       c=df.ix[start+index*2:start+index*3-1,:]
       d=df.ix[start+index*3:,:]
       
       a.index=range(len(a))
       b.index=range(len(b))
       c.index=range(len(c))
       d.index=range(len(d))
       return a,b,c,d
 
def coords_trans_baidu100(data,key = key):

    
    url_A = 'http://api.map.baidu.com/geoconv/v1/?coords='
    url_C = '&from=3&to=5&ak=' + key
    url_res = []
    json_res= []
    lat_res = []
    lon_res = []
    np_baidu_lat = np.array(data['lat'],dtype='float64')
    np_baidu_lon = np.array(data['lon'],dtype='float64') 
    xy = [(x,y) for x,y in zip(np_baidu_lat,np_baidu_lon)]
    length = len(data)

    progress = progressbar.ProgressBar(max_value=math.ceil(length//100)).start()
    
    
    try:
        xy_li = [xy[i:i+100] for i in range(0,length,100)] 
        for i,xy_tuple_li in enumerate(xy_li):
            
            str_li = [','.join([str(xy_tuple[0]),str(xy_tuple[1])]) for xy_tuple in xy_tuple_li]
            
            str_join = ';'.join(str_li)
            url = url_A+str_join+url_C
            
            try:
                resjson = json.loads(urlopen(url).read().decode('utf-8'))
                time.sleep(0.01)
                progress.update(i)
                json_res.append(resjson)
                url_res.append(url)
                
                lat_arr, lon_arr = json_parse(resjson) #返回解析每一个resjson的lat和lon数组
                
                lat_res.append(list(lat_arr))
                lon_res.append(list(lon_arr))
                
            except Exception as e:
                print('******远程主机强制关闭或者RemoteDisconnected,i:%s*******'%i)
                url_res.append('error')
                lat_res.append(np.array([]))
                lon_res.append(np.array([]))
    except Exception as e:
        print(e)
    finally:
        df_json = pd.DataFrame(json_res)
        df_json.to_csv(city+'json_res.csv')
        df_url = pd.DataFrame(url_res,columns=['urls'])
        df_url.to_csv(city+'url_res.csv')
        
        
    progress.finish()
    latresult = [i for k in lat_res for i in k]
    lonresult = [i for k in lon_res for i in k]
    df_baidu_lat_lon = DataFrame([])
    df_baidu_lat_lon['baidu_lat'] = latresult
    df_baidu_lat_lon['baidu_lon'] = lonresult
    df_baidu_lat_lon.to_csv(city+'baidu_lat_lon_BSV_id.csv') 
    print('Done!')
    return url_res,json_res,latresult,lonresult


def json_parse(resjson):
    if resjson and resjson['status'] == 0:
        lat_arr = np.zeros(len(resjson['result']),dtype='float64')
        lon_arr = np.zeros(len(resjson['result']),dtype='float64')
        for i,json in enumerate(resjson['result']):
            LAT = json['x']
            LON = json['y']
            lat_arr[i] = LAT
            lon_arr[i] = LON
        return lat_arr,lon_arr
    else:
        return np.array([]),np.array([])
        

def baidu_crawler_v2(data): 
    startwith = data['ID'][0]
    print(startwith)
    data_path = r'E:\project_y\data'
    err_li = []
    response_li = []
    url_A = 'http://api.map.baidu.com/panorama/v2?'
    url_B = 'ak=8ljimGEfgienvbpHSDFOBag5ISN3K49&width=1000&height=500&fov=360' 
    progress = progressbar.ProgressBar(max_value=len(data)).start()
    for i in range(len(data)):
        flag = str(startwith+i)
        LAT = data.iloc[i]['baidu_lat']
        LON = data.iloc[i]['baidu_lon']
        #print(LAT,LON)
        progress.update(i)
        folder_path = r'E:/project_y/Beijing_BSVs/' 
        panoid = 'BSV_'+city+flag 
        name = 'BSV_'+city+str(startwith+i)+'.png'
        time.sleep(0.01) #防止服务器误认为是网络攻击
        url_C = '&location='+str(LAT)+','+str(LON)
        url = url_A+url_B+url_C
        try: 
            if os.path.isfile(os.path.join(folder_path,name)):
                continue
            conn = urlopen(url) 
            try:
                res = conn.read() 
                response = json.loads(res.decode('utf-8')) 
                response_li.append((flag,url,response)) 
            except (json.decoder.JSONDecodeError, UnicodeDecodeError) as e: 
                f = open(os.path.join(folder_path,name),'wb')
                f.write(res)
                f.close()
        except Exception as e:
            print('******远程主机强制关闭或者RemoteDisconnected,i:%s*******'%flag) 
            err_li.append(url) 
            print(e)

    progress.finish()
    
    ti = time.localtime()
    now = str(ti.tm_year)+str(ti.tm_mon)+str(ti.tm_mday)+str(ti.tm_hour)+str(ti.tm_min)
    df_err = DataFrame(Series(err_li),columns=['err_list'])
    df_err.to_csv('err_list_crawling_'+now+'.csv') 
    df_response = DataFrame(Series(response_li))
    df_response.to_csv('JSON_no_pano'+now+'.csv') 
    print('Done!') 
    
    return err_li 


def baidu_crawler_v3(data,key = key30,folder_path = folder_path):
    data = data.set_index('FID')
    siren = data['panoid'] # Series; index is FID, value is panoid
    err_li = []
    response_li = []
    url_A = 'http://api.map.baidu.com/panorama/v2?'
    url_B = 'ak='+key+'&width=1000&height=500&fov=360' # 在家用key30
    progress = progressbar.ProgressBar(max_value=len(data)).start()
    for i,si in enumerate(siren.index): # 用FID迭代
        LAT = data.ix[si]['baidu_lat']
        LON = data.ix[si]['baidu_lon']
        progress.update(i)
        name = str(siren[si])+'.png'
        time.sleep(0.2) 
        url_C = '&location='+str(LAT)+','+str(LON)
        url = url_A+url_B+url_C

        try: 
            if os.path.isfile(os.path.join(folder_path,name)):
                continue
            conn = urlopen(url) 
            try:
                res = conn.read()
                response = json.loads(res.decode('utf-8')) 
                response_li.append((flag,url,response)) 
            except (json.decoder.JSONDecodeError, UnicodeDecodeError) as e: 
                f = open(os.path.join(folder_path,name),'wb')
                f.write(res)
                f.close()
        except Exception as e:
            print('******远程主机强制关闭或者RemoteDisconnected,i:%s*******'%flag) 
            err_li.append(url) 
            print(e)
        
    progress.finish()

    ti = time.localtime()
    now = str(ti.tm_year)+str(ti.tm_mon)+str(ti.tm_mday)+str(ti.tm_hour)+str(ti.tm_min)
    df_err = DataFrame(Series(err_li),columns=['err_list'])
    df_err.to_csv('err_list_crawling_'+now+'.csv')
    df_response = DataFrame(Series(response_li))
    df_response.to_csv('JSON_no_pano'+now+'.csv') 
    print('Done!') 

    return err_li 

'''
Pool function
'''
def pool_processing(f,df_list):
	p = Pool(4)
	results = p.map(f,df_list)
	p.close()
	p.join()
	return results

def panoid_ge(data, folder_path = folder_path):
    panoid_list = []
    for i,da in enumerate(data['FID']):
        panoid = 'BSV_'+city+str(da)
        panoid_list.append(panoid)

    data['panoid'] = panoid_list
    data.to_csv(city+'baidu_lat_lon_with_panoid.csv') 
    return data


if __name__ == '__main__':
    
    
    
    res = coords_trans_baidu100(data) 
    df = pd.read_csv(city+'baidu_lat_lon_BSV_id.csv') 
    df.columns = ['FID','baidu_lat','baidu_lon']
    

    df = pd.merge(data,df,on=['FID'],how='inner')
    df_panoid = panoid_ge(df)
    df_panoid.to_csv(city+'baidu_lat_lon_with_panoid.csv')
    
	
    
    df = pd.read_csv(city+'baidu_lat_lon_with_panoid.csv')
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    start = 0
    dfdf = df[start : ] 
    df1, df2, df3, df4 = div_df4(dfdf, start = start)
    df_list = [df1, df2, df3, df4]
    results = pool_processing(baidu_crawler_v3,df_list)
    print ('Done!')
    print (results)
    