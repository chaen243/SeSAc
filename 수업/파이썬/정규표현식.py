# import re
# # '.' 한개의 임의의 문자를 나타냄
# # r = re.compile('a.c') #a와 c 사이에 어떤 1개의 문자라도 올 수 있는것.
# # print(r.search('kkk')) #출력 없음
# # print(r.search('abc')) # span=(0, 3),<-찾고자 하는 문자열은 3글자 match='abc'
# # print(r.search('abbbbbbbc')) #출력 없음으로 뜸

# # # '?': ?앞의 문자가 존재할 수도 있고, 존재하지 않을 수도 있는 경우를 나타냄.
# # r = re.compile('ab?c')
# # print(r.search('abbc')) #출력 없음
# # print(r.search('abc')) #b가 있는것을 판단하여 abc를 매치함
# # print(r.search('ac'))

# # '*': 바로 앞의 문자가 0개 이상 있을 경우 출력이 됨

# r = re.compile('ab*c')
# print(r.search("a")) #출력 없음
# print(r.search("ac")) #span=(0, 2), match='ac'
# print(r.search('abbc')) #span=(0, 4), match='abbc'>

# # '+': *와 유사함. but 앞의 문자가 최소 1개 이상인 경우 작동
# r = re.compile('ab+c')
# print(r.search('ac')) #출력없음
# print(r.search('abc')) #span=(0, 3), match='abc'>
# print(r.search('acc')) #출력없음

# # '^': 시작되는 글자를 지정
# r = re.compile('^a')
# print(r.search('bbc')) #none
# print(r.search('ab')) #span=(0, 1), match='a'>

# #[] : 문자들을 넣으면 그 문자들 중 한개의 문자와 매치한다는 의미를 가짐.
# #[a-c]=[abc]
# #[0-5]=[012345]
# #[a-zA-Z]#모든 알파벳

# #{숫자} : 숫자갯수만큼 출력
# text = '전화번호는 010-1234-5678'
# pat = re.compile("\d{3}-\d{4}-\d{4}")
# phone_list = pat.findall(text)
# print(phone_list)

# pat1 = re.compile("[0-9]{3}-[0-9]{4}-[0-9]{4}")
# phone_list1 = pat1.findall(text)
# print(phone_list1)


#이진수 십진수로

#1111

