# -*- coding:utf-8 -*-
import csv
import json
import codecs
import os
import string
import sys
import time
import urllib.request
from urllib.parse import quote
from builtins import object, float, range, int, len, open, list, str


class BaiDuPOI(object):
    def __init__(self, itemy, loc):
        self.itemy = itemy
        self.loc = loc

    def urls(self):
        api_key = baidu_api
        urls = []
        for pages in range(0, 2):
            url = 'http://api.map.baidu.com/place/v2/search?query=' + self.itemy + '&bounds=' + self.loc + '&page_size=20&page_num=' + str(
                pages) + '&output=json&ak=' + api_key
            urls.append(url)
        return urls

    def baidu_search(self):
        '''json_sel = []
        for url in self.urls():
            s = quote(url, safe=string.printable)
            json_obj = urllib.request.urlopen(s).read().decode('utf-8')
            data = json.loads(json_obj)
            for item in data['results']:
                jname = item["name"]
                jlat = item["location"]["lat"]
                jlng = item["location"]["lng"]
                js_sel = jname + ',' + str(jlat) + ',' + str(jlng)
                json_sel.append(js_sel)
        return json_sel
        '''
        data = []
        for url in self.urls():
            s = quote(url, safe=string.printable)
            json_obj = urllib.request.urlopen(s).read().decode('utf-8')
            data.append(json.loads(json_obj))
            time.sleep(1)  # 休眠1秒
        return data


class LocaDiv(object):
    def __init__(self, loc_all):
        self.loc_all = loc_all

    def lat_all(self):
        lat_sw = float(self.loc_all.split(',')[0])
        lat_ne = float(self.loc_all.split(',')[2])
        lat_list = []
        for i in range(0, int((lat_ne - lat_sw + 0.0001) / 0.2)):  
            lat_list.append(lat_sw + 0.2 * i)  # 0.05
        lat_list.append(lat_ne)
        return lat_list

    def lng_all(self):
        lng_sw = float(self.loc_all.split(',')[1])
        lng_ne = float(self.loc_all.split(',')[3])
        lng_list = []
        for i in range(0, int((lng_ne - lng_sw + 0.0001) / 0.3)):  
            lng_list.append(lng_sw + 0.3 * i)  
        return lng_list

    def ls_com(self):
        l1 = self.lat_all()
        l2 = self.lng_all()
        ab_list = []
        for i in range(0, len(l1)):
            a = str(l1[i])
            for i2 in range(0, len(l2)):
                b = str(l2[i2])
                ab = a + ',' + b
                ab_list.append(ab)
        return ab_list

    def ls_row(self):
        l1 = self.lat_all()
        l2 = self.lng_all()
        ls_com_v = self.ls_com()
        ls = []
        for n in range(0, len(l1) - 1):
            for i in range(0 + len(l1) * n, len(l2) + (len(l2)) * n - 1):
                a = ls_com_v[i]
                b = ls_com_v[i + len(l2) + 1]
                ab = a + ',' + b
                ls.append(ab)
        return ls


if __name__ == '__main__':
    doc = open('NTPOI.csv', 'a+')
    writer = csv.writer(doc)

    # ak
    baidu_api = "*********************"  # baidu key
    print("crawling...")
    start_time = time.time()
    loc = LocaDiv('31.69,120.54,32.65,121.95')
    locs_to_use = loc.ls_row()

    for loc_to_use in locs_to_use:
        par = BaiDuPOI('学校', loc_to_use)  # category of POI, 学校 means school in Chinese
        '''
        a = par.baidu_search()
        for ax in a:
            writer.writerow(a)
       '''

        listdata = par.baidu_search()
        for resultIndex in range(len(listdata)):  
            resultlist=listdata[resultIndex]["results"]
            for pIndex in  range(len(resultlist)):
                 writer.writerow(list(resultlist[pIndex].values()))
        # time.sleep(1)  # 休眠1秒

    doc.close()
    end_time = time.time()
    print("done, time:%.2fsecond" % (end_time - start_time))