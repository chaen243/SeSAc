from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By #find_element 함수를 쉽게 쓰기위함.
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import time

sys.stdout.reconfigure(encoding = 'utf-8')

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("lang=ko_KR")



 

#lists = driver.find_elements(By.CLASS_NAME,'viewtype') #5viewtype catal_ty : 5개를 가지고 옴
#print(len(lists))

scroll_pause_sec = 1
last_height = driver.execute_script('return document.body.scrollHeight')

while True:
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')

    #1초 대기
    time.sleep(scroll_pause_sec)

    #스크롤 다운 후 스크롤 높이를 다시 가져옴
    new_height = driver.execute_script('return document.body.scrollHeight')
    if new_height == last_height:
        break
    last_height = new_height


lists = driver.find_element(By.ID,'bestPrdList').find_elements(By.CLASS_NAME,'viewtype')
#bestProdList는 1개이고, viewtype(viewtype catal_ty)는 여러개가 있음

#li 태그 가지고 옴
for list in lists:
    bestlist = list.find_elements(By.TAG_NAME,'li') #리스트 전부 들고옴
    for item in bestlist:
        print('No : ', item.find_element(By.CLASS_NAME, 'best').text) #best = span함수에 묶여있는 순위
        print('Product : ', item.find_element(By.CLASS_NAME, 'pname').find_element(By.TAG_NAME, 'p').text) #제품명
        print('Price : ', item.find_element(By.CLASS_NAME, 'sale_price').text) #가격
        print('URL : ',item.find_element(By.CLASS_NAME, 'box_pd.ranking_pd').find_element(By.TAG_NAME, 'a').get_attribute('href')) #url
        # # print('image :', item.find_element(By.CLASS_NAME, 'img_plot').find_element(By .TAG_NAME, 'img').get_attribute(By.TAG_NAME, 'src') #이미지 접근
        # #urllib.requests.urlretrieve(url, '')
        # print('-'*100)



