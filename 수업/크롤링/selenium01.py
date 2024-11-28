
'''
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By #find_element를 쉽게 쓰기 위함.
#url 꺼짐 방지
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

driver = webdriver.Chrome(options=chrome_options)
driver.get('https://www.google.com')
# driver.find_element(By.CLASS_NAME, "gLfyf").send_keys("아이유")
# driver.find_element(By.CSS_SELECTOR, "center > input.gN089b").click()

driver.find_element(By.NAME, 'q').send_keys("아이유")
driver.find_element(By.NAME, 'q').submit()
driver.find_element(By.NAME, 'btnk').click()
'''
'''
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options()
chrome_options.add_argument("--lang=ko_KR")#한국어
chrome_options.add_experimental_option("detach",True)

driver = webdriver.Chrome(options = chrome_options)
driver.get('https://www.google.com')
search_box = driver.find_element(By.NAME, 'q')
search_box.send_keys("아이유")
search_button = WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.NAME,'btnK')))
search_button.click()


# driver.find_element(By.CLASS_NAME,"gLFyf").send_keys("아이유")
# driver.find_element(By.CLASS_NAME,"gLFyf").send_keys(Keys.ENTER)

driver.find_element(By.XPATH,'//*[@id="hdtb-sc"]/div/div/div[1]/div/div[3]/a').click()
driver.execute_script('document.charset = "UTF-8";')
titles = []
links_1 = []
links = driver.find_elements(By.CSS_SELECTOR, 'div > div > a > div > div.SoAPf >div.n0jPhd.ynAwRc.MBeuO.nDgy9d')
# for link in links:
#     titles.append(link.text)
# print(titles)    

for link in links:
    titles.append(link.text)
print(titles)

'''
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By #find_element 함수를 쉽게 쓰기위함.
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys

sys.stdout.reconfigure(encoding = 'utf-8')

chrome_options = Options()
chrome_options.add_experimental_option("detach",True)
chrome_options.add_argument("--lang=ko_KR")
driver =webdriver.Chrome(options=chrome_options)
driver.get('https://www.google.com')
search_box = driver.find_element(By.NAME,'q')
search_box.send_keys('아이유')

search_button = WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.NAME,'btnK')))
search_button.click()

driver.find_element(By.XPATH,'//*[@id="hdtb-sc"]/div/div/div[1]/div/div[3]/a').click()

titles = []
links = driver.find_elements(By.CSS_SELECTOR, ".n0jPhd.ynAwRc.MBeuO.nDgy9d")
for link in links:
    titles.append(link.text)
print(titles)