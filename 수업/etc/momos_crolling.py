#크롤링 숙제
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

url = 'https://momos.co.kr/custom/sub/product_category/sd_shop_roasted_bean.html'

driver = webdriver.Chrome(options= chrome_options)
driver.get(url)

link_selector = '//*[@id="contents"]/div[2]/nav/ul/li[2]/a'
link_element = driver.find_element(By.XPATH,link_selector)
link_element.send_keys('\n') #내부명령어를 받지 못해 \n으로 강제로 엔터를 넣어줌

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


# time.sleep(3)

product_container = driver.find_element(By.ID, 'contents')
product_list = product_container.find_elements(By.CLASS_NAME, 'prdList01') 

    

# for list_item in product_list:
#     items = list_item.find_elements(By.TAG_NAME, 'li')
#     spans = items.find_elements(By.TAG_NAME, 'span')
    
#     for item, span in items:
#         if 'font-size:12px;color:#555555;' in span.get_attribute('style'):
#             Price = span.text
#         print('Product : ', item.find_element(By.CLASS_NAME, 'name').text)
#         print('Taste : ', item.find_element(By.CLASS_NAME, 'xans-record-').text)
#         # print('Price : ', item.find_element(By.XPATH, '//*[@id="anchorBoxId_82"]/div[2]/ul/li[2]').text)
        
#         print('Price : ', Price)        

try:
    # 상품명
    product_name = driver.find_element(By.CSS_SELECTOR, 'div.name a span').text
    print("상품명:", product_name)
    
    # 설명
    product_desc = driver.find_element(By.CSS_SELECTOR, 'ul.spec li span').text
    print("설명:", product_desc)
    
    # 가격
    product_price = driver.find_element(By.CSS_SELECTOR, 'li[data-title="판매가"] span').text
    print("가격:", product_price)

finally:
    # 브라우저 닫기
    driver.quit()