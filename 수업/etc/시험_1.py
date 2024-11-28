# 문제 1) 
# 십진수 a와 b가 주어집니다.
# 주어진 십진수 a와 b를 이진수로 변환하고, 그 후 다음 연산을 수행하세요:
#  결과를 이진수로 출력하세요.

#10진수를 2진수로 변환
#2진수는 10진수를 2로 나눈 나머지의 배열
def d_to_b(n):  
    binary = '' #빈 문자열 생성
    while n>0:  #n이 0보다 클때까지 반복
        r = n%2 #r은 n 나누기 2의 나머지이다. (이진수의 자리수)
        binary = str(r)+binary #binary에 문자형인 r과 이전 binary를 더해서 이진수를 만들어줌
        n = n//2 #n은 n나누기 2의 몫이다 (while문이 돈 뒤에 다시 돌때 n을 바꿔주기 위해.)
    return binary    #만들어진 이진수를 반환

print(d_to_b(73))
print(d_to_b(70))
# a와 b를 더한세요.
print(d_to_b(73+7))
# a에서 b를 뺀 결과를 이진수로 출력하세요.
print(d_to_b(73-7))
# a와 b의 AND 연산 결과를 이진수로 출력하세요.
print(d_to_b(73 & 7))
# a와 b의 OR 연산 결과를 이진수로 출력하세요.
print(d_to_b(73 | 7))
# a와 b의 XOR 연산 결과를 이진수로 출력하세요
print(d_to_b(73 ^ 7))



#10진수 -> 5진수 변환
#10진수에서 2진수를 변환하는 함수와 숫자만 바꿔서 진행

def t_to_f(a):
    five = ''
    while a>0:
        r = a%5
        five = str(r)+five
        a = a//5
    return five 

print(t_to_f(123))    





# 문제2)
# 2진수를 5진수와 10진수로 바꾸는 함수를 작성하시오.


#2진수를 10진수로

def t_to_ten(b):
    bb = b[::-1] #입력된 이진수를 뒤집어서 뒤에서부터 수를 세어서 십진수로 변환
    ten = 0 #변수 초기화

    #순차적으로 처리하기 위해 for반복문 사용
    for i in range(len(bb)): 
        ten += int(bb[i])*2**i  #ten변수에 2의 거듭제곱 값들을  더함
    return ten    

binary_str = '111101'
print(t_to_ten(binary_str))    
    
#10진수가 주어지면 2진수와 5진수로 바꿀수 있는 한꺼번에 작동할 수 있는 함수를 작성하시오.

def change_numbers():
    i = input("2진수는 2, 10진수는 10을 입력하세요: ")

    if i == '2': #입력된 수가 2이면
        binary = input("2진수를 입력하세요: ") #2진수를 입력
        try:
            decimal = t_to_ten(binary)  # 2진수를 10진수로 변환하고
            five = t_to_f(decimal)  # 10진수를 5진수로 변환
            print(f"입력된 2진수 {binary}는 10진수로 {decimal}이고 5진수로 {five}입니다.")
        except ValueError: #valueError가 날땐 종료
            print("잘못된 수입니다.")
    
    elif i == '10': 
        decimal = int(input("10진수를 입력하세요: "))
        binary = d_to_b(decimal)  # 10진수 입력시 이진수로 변환
        five = t_to_f(decimal)  # 10진수를 5진수로 변환
        print(f"입력된 10진수 {decimal}는 2진수로 {binary}이고 5진수로 {five}입니다.")
    else:
        print("잘못된 수입니다.")

change_numbers()




# (파이썬)


# 문제 4 : 문자열 속 단어 수 세기
# 목표: 사용자가 입력한 문자열에서 단어의 개수를 세는 프로그램을 작성합니다.
# 게임 규칙:
# 사용자는 한 줄의 문자열을 입력합니다.
# 프로그램은 그 문자열에서 공백을 기준으로 단어를 분리하고, 단어의 개수를 출력합니다.
# 공백만 있는 문자열이나, 앞뒤 공백이 있을 경우에도 올바르게 처리해야 합니다.


def count_words(a):#입력받은 문자열의 갯수를 세어주는 함수
    words= a.split() #split()으로 문자열을 공백 기준으로 분리해 리스트로 반환
    return len(words) #분리된 단어 갯수를 반환

input_s = input('문자열을 입력하세요:')

print(count_words(input_s))



# 문제 5 : 숫자 추측 게임
# 목표: 컴퓨터가 1부터 주어진 범위 내에서 무작위로 숫자를 선택하고, 사용자는 그 숫자를 추측해야 합니다. 사용자가 추측할 때마다 컴퓨터는 힌트를 제공합니다. 힌트는 "더 작은 숫자입니다." 또는 "더 큰 숫자입니다."로 제공되며, 정답을 맞출 때까지 계속 반복됩니다.
# 게임 규칙:
# 컴퓨터가 범위 내에서 무작위로 숫자를 선택합니다.
# 사용자는 숫자를 입력하여 추측합니다.
# 매번 추측 후, 컴퓨터는 힌트를 제공합니다:
# o    "더 작은 숫자입니다." 또는 "더 큰 숫자입니다."
# o    정답을 맞추면 게임이 종료됩니다.
# 사용자가 정답을 맞출 때까지 계속 추측하며 점수를 받습니다.
# 점수는 100점으로 시작하고, 틀릴 때마다 10점씩 차감됩니다.


import random #랜덤하게 숫자를 가져오기위해 random 패키지 불러옴


num = random.randint(1,101) # 랜덤한 정수 1~100사이 범위 지정
score = 100 #시작 점수 100점 지정

a = 99999 #while문 조건 지정을 위해 임의의 수 지정
while num!=a: #num과 a가 같지 않으면 while문 반복 진행
    a = int(input('숫자를 입력하세요 :')) #추측할 입력 숫자 지정
    
    if a < num:
        print('정답은 더 큰 숫자입니다') 
    elif a > num:
        print('정답은 더 작은 숫자입니다.')
    elif a == num:
        print('정답입니다 점수는 %d 점 입니다'%score)    #score는 추측횟수마다 달라지기때문에 %d(정수)와 %score(지정한변수) 사용 
        break #정답이면 while문을 끝내기 위해 break 사용
    score -= 10 #while문이 한바퀴 돌때마다 10점씩 차감하기 위해 score -10 지정







