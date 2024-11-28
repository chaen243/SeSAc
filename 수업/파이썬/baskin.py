import random

print('베스킨라빈스 31 게임')
print('1부터 31까지의 숫자를 번갈아 불러 31을 부르는 사람이 지는 게임 입니다.')

number = 0 #변수 초기화

turn = 0  #횟수 변수 초기화

while True:
    if turn == 0: #순서가 0일때 
        p1 = int(input('p1 부를 숫자의 개수를 입력하세요 (1 ~ 3): ')) #시작할 숫자 지정
        for _ in range(p1): #p1의 갯수만큼  반복
            number += 1 #숫자 더함.
            print('p1:', number) #숫자 출력

        turn += 1 #turn = turn+1
        turn %= 2 #turn = turn%2
        
    elif turn == 1:   
        p2 = random.randint(1, 3)
        for _ in range(p2):
            number += 1
            print('p2', number)

        turn += 1
        turn %= 2

    if number >= 31: 
        break

if turn == 0:
    print('p1 승리')
else:
    print('p2 승리')