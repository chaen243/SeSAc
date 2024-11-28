n = int(input('숫자:'))

def div(n):
    result = []
    while n:
        result+=(str(n%2))
        n//=2
    return ''.join(result)

print(div(n))    
