#최빈값 구하기

arr_1 = [3,4,7,7,1,5,4,2,7,3,3,5,1,3,4]
      
def solution(array):
    count = [0] * (max(array)+1) 

    for i in array:
        count[i] += 1 
    m = 0 
    for c in count:  
        if c == max(count):
            m +=1

    if m >1 :
        answer = -1
    elif m == 1:
        answer = count.index(max(count)) 

    return answer    

print(solution(arr_1))



#매니저님 코드
#어레이 딕셔너리로 바꾸기
#딕셔너리 반복으로 최빈값 구하기
#최빈값이 여러개인지 확인.
# dict = {}
def sol(array):
    dict = {}
    for i in array:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1
       
    max_1 = 0   
    lst = []

    for i,j in dict.items():
        if j> max_1:
            max_1 = j
            lst = [i]
        elif j == max_1:
            lst.append(i)

    if len(lst) ==1:
        return lst[0]   
    else:
        return -1         
    
print(sol(arr_1))    
            
