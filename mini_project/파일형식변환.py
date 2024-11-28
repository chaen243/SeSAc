import numpy as np

input_file = "C:\\Users\\r2com\\Desktop\\수업자료\\mini_project\\sanitation.csv"  # 한글 인코딩된 원본 파일
output_file = "C:\\Users\\r2com\\Desktop\\수업자료\\mini_project\\rsanitation_utf8.csv"  # UTF-8로 변환된 파일


# import chardet
# with open(input_file, 'rb') as f:
# rawdata = f.read()
# result = chardet.detect(rawdata)
# print(result)  # {'encoding': 'utf-8', 'confidence': 0.99}
#파일 변환
with open(input_file, "r", encoding="EUC-KR") as infile:
    content = infile.read()  # 파일 읽기

#UTF-8로 저장
with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.write(content)

print("파일이 UTF-8 형식으로 저장되었습니다:", output_file)
