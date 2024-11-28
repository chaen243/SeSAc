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

driver = webdriver.Chrome(options=chrome_options)
driver.get(url)

# 페이지 내 링크 클릭
WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, '//*[@id="contents"]/div[2]/nav/ul/li[2]/a'))
).click()

# 상품 리스트 추출
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, 'prdList'))
)

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

lists = driver.find_elements(By.CLASS_NAME, 'prdList')

for list in lists:
    product_items = list.find_elements(By.TAG_NAME, 'li')
    for item in product_items:
        product_name = item.find_element(By.CLASS_NAME, 'name').text
        print('Product:', product_name)
