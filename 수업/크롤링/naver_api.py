import os
import sys
import urllib.request

'''
client_id = "RzKVFU8prLxgz7d3lPh1"
client_secret = "8bBQOSp0FF"
encText = urllib.parse.quote("경복궁")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
'''

Naver_client_id = 'Rs6MKVFU8prLxgz7d3lPh1'
Naver_client_secret = '8bBQOSp0FF'
Kakao_API_key= '39a2147637c0f4e5a936fbee34aecb77a' #인증받은 rest api key 값 입력할 것.
Google_SEARCH_ENGINE_ID = '05bad5297035d417c'
Google_API_KEY = 'AIzaSyCBNzYjzd6gs8cRnnQZe6FOgYzusrLL6Xo'
    
Trash_Link = ["tistory", "kin", "youtube", "blog", "book", "news", "dcinside", "fmkorea", "ruliweb", "theqoo", "clien", "mlbpark", "instiz", "todayhumor"]

from datetime import datetime
import os
import sys
import urllib.request
import pandas as pd
import json
import re
import requests
import simplejson

def Google_API(query, wanted_row):
    query= query.replace("|","OR")
    query += "-filetype:pdf"
    start_pages=[]

    df_google= pd.DataFrame(columns=['Title','Link','Description'])

    row_count =0


    for i in range(1,wanted_row+1000,10):
        start_pages.append(i)

    for start_page in start_pages:
        url = f"https://www.googleapis.com/customsearch/v1?key={Google_API_KEY}&cx={Google_SEARCH_ENGINE_ID}&q={query}&start={start_page}"
        data = requests.get(url).json()
        search_items = data.get("items")

        try:
            for i, search_item in enumerate(search_items, start=1):
                # extract the page url
                link = search_item.get("link")
                if any(trash in link for trash in Trash_Link):
                    pass
                else:
                    # get the page title
                    title = search_item.get("title")
                    # page snippet
                    descripiton = search_item.get("snippet")
                    # print the results
                    df_google.loc[start_page + i] = [title,link,descripiton]
                    row_count+=1
                    if (row_count >= wanted_row) or (row_count == 300) :
                        return df_google
        except:
            return df_google
    return df_google