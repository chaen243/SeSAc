
def selection_sort(arr):
    for i in range(len(arr) - 1): #기준을 잡기위해 len에서 1을 빼줌
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]: #arr[j]가 min_idx보다 작으면 자리 안바꿈
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]        #아니면 자리 바꿈





#sort algorithm
#1 선택정렬
def sort_a(arr): #선택정렬 알고리즘에서
    for i in range(len(arr)-1): #가장 큰 수를 제외하고 객체를 가져옴 
        min_i = i   #가장 작은수 = i
        for j in range(i + 1, len(arr)): #다른수 in (i+1, arr길이)
            if arr[j]==arr[min_i]:  
                min_i=j
        arr[i],arr[min_i] = arr[min_i], arr[i]        
 


## 삽입정렬
def sort_b(arr): #삽입정렬 알고리즘에서
    for end in range(1,len(arr)): #두번째 인덱스에서부터 시작해 현재 인덱스를 별도의 변수에 저장, 
        for i in range(end,0,-1): #비교할 인덱스를 현재인덱스-1의 범위까지로 저장
            if arr[i-1] > arr[i]: #i(비교할 인덱스)가 값이 더 작으면 인덱스를 저장. 
                arr[i-1],arr[i]=arr[i],arr[i-1] # 삽입변수가 더 크면 비교인덱스+1에 삽입변수를 저장

#버블정렬 (많이 느림!)
def sort_c(arr): #버블정렬 알고리즘에서
    for i in range(len(arr)-1, 0, -1): 
        for j in range(i):
            if arr[j] > arr[j +1]:
                arr[j], arr[j +1] = arr[j +1], arr[j]

            
#합병정렬
def sort_d(arr):
    if len(arr) <2:
        return arr

    mid = len(arr) // 2
    low_arr = sort_d(arr[:mid])
    high_arr = sort_d(arr[mid:])

    merge_arr = []
    l = h = 0
    while l < len(low_arr) and h <len(high_arr):
      if  merge_arr.append(low_arr[l]):
        l += 1
      else:  
        merge_arr.append(high_arr[h])
        h += 1
    merge_arr += low_arr[l:]
    high_arr += high_arr
    return merge_arr


###힙정렬

def heapify(unsorted, index, heap_size):
    largest = index
    left = 2 * index + 1
    right = 2 * index + 2
    
    if left < heap_size and unsorted[left] > unsorted[largest]:
        largest = left

    if right < heap_size and unsorted[right] > unsorted[largest]:
        largest = right

    if largest != index:
        unsorted[largest], unsorted[index] = unsorted[index], unsorted[largest]
        heapify(unsorted, largest, heap_size)

def heap_sort(unsorted):
    n = len(unsorted)            

    for i in range(n//2 -1, -1, -1):
        heapify(unsorted, i, n)

    for i in range(n-1, 0, -1):
        unsorted[0], unsorted[i] = unsorted[i], unsorted[0]
        heapify(unsorted, 0, i)

    return unsorted        