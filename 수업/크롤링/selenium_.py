import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import requests
from bs4 import BeautifulSoup

# Colab에서 사용하는 Chrome 및 Chromedriver 경로 설정
chrome_bin_path = "/usr/bin/chromium-browser"
chromedriver_path = "C:\\Users\\r2com\\Downloads\\chromedriver-win64"

# Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")  # GUI 없이 실행
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.binary_location = chrome_bin_path

# Selenium 드라이버 설정
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# 연도별 이미지 스크래핑 및 다운로드 코드
for year in range(2018, 2023):
    url = f"https://search.daum.net/search?w=tot&q={year}%EB%85%84%20%EC%98%81%ED%99%94%20%EC%88%9C%EC%9C%84"
    driver.get(url)
    time.sleep(3)  # 페이지 로딩 시간 대기

    # BeautifulSoup을 사용하여 페이지 소스를 분석
    soup = BeautifulSoup(driver.page_source, "lxml")
    images = soup.find_all("img", class_="thumb_img")  # 필요한 경우 클래스명 수정

    if not images:
        print(f"{year}년에는 이미지가 없습니다.")
    else:
        count = 0
        for idx, img_tag in enumerate(images):
            image_url = img_tag.get("src") or img_tag.get("data-src")

            # base64 인코딩된 이미지나 URL이 없을 경우 건너뛰기
            if not image_url or image_url.startswith("data:image/"):
                continue

            # //로 시작하는 URL에 https 추가
            if image_url.startswith("//"):
                image_url = "https:" + image_url

            try:
                # 이미지 다운로드
                image_res = requests.get(image_url)
                image_res.raise_for_status()

                with open(f"movie_{year}_{idx + 1}.jpg", "wb") as f:
                    f.write(image_res.content)

                count += 1
                if count >= 5:  # 연도별로 최대 5개의 이미지만 다운로드
                    break

            except requests.exceptions.HTTPError as e:
                print(f"{year}년의 이미지 {idx + 1} 다운로드 실패: {e}")
            except Exception as e:
                print(f"{year}년의 이미지 {idx + 1}에서 오류 발생: {e}")

# 드라이버 닫기
driver.quit()