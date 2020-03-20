import numpy as np

# numpy.c_(column bind)와 numpy.r_ (row bind)의 비교
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a, type(a), a.shape)
print(b, type(b), b.shape)

c = np.c_[a, b]
print(c, type(c), c.shape)

d = np.r_[a, b]
print(d, type(d), d.shape)


e = np.array([[1, 2, 3],
             [4, 5, 6]])
f = np.array([[10, 20],
              [30, 40]])

# col 개수가 안맞아서 row를 못 붙임
print(np.c_[e, f]) # e와 f의 row 개수가 같아야 column 방향으로 붙일 수 있음
# print(np.r_[e, f]) # e와 f의 column 개수가 다르면 row방향으로 붙일 수 없음
# ValueError: all the input array dimensions for the concatenation axis must match exactly,
# but along dimension 1, the array at index 0 has size 3 and the array at index 1 has size 2
# 매우 specific


g = np.array([[100, 200, 300]]) # row의 개수가 다르기 때문에 오른쪽으로 붙일 수 없음.
#print(np.c_[e, g]) # row의 개수가 다르기 때문에 오른족으로 붙일 수 없음!
print(np.r_[e, g]) # col의 개수가 같아야 밑으로 붙일 수 있음

# 같은 차원끼리 사용

# numpy에 관해서 좀 얘기해보야여?
# numpy에 여러가지 array들을 만들어낼 수 있는 함수가 있다


# 1. (2, 3) shape의 모든 원소가 1인 array를 생성해서 출력 : A
A = np.ones((2, 3), dtype=np.int) # int로 변환시켜주려면 dtype 추가
print(A)

# 2. (2, 3) shape의 모든 원소가 0인 array를 생성해서 출력 : B
B = np.zeros((2, 3))
print(B)

# 3. (3, 2) shape의 원소는 1~ 6인 array를 생성해서 출력 : C
C = np.array(range(6)).reshape((3, 2))
print(C)
print(np.arange(1, 7).reshape((3, 2)))

# 4. (3, 2) shape의 난수들로 이루어진 array를 생성해서 출력 : D
D = np.random.random((3, 2))
print(D)


"""다음과 같은 결과가 나올 수 있도록 
numpy를 사용하지 않고 1) add(x, y), 2) subtract(), 3) multiply(), 4) divide(), 5) dot() 함수를 구현
|1 2| + |5 6|= |6  8 | 
|3 4|   |7 8|  |10 12|

|1 2| - |5 6|= |-4 -4| 
|3 4|   |7 8|  |-4 -4|

|1 2| * |5 6|= |5  12| 
|3 4|   |7 8|  |21 32|

|1 2| / |5 6|= |0.2   0.333| 
|3 4|   |7 8|  |0.428 0.5  |

|1 2| @ |5 6|= |19 22| 
|3 4|   |7 8|  |43 50|
위의 결과와 같은 결과를 주는 numpy 코드를 작성
"""
#1 add(x, y)

def add(x, y):
    ''' 행렬끼리 더해주는 함수, X도 행렬 y도 행려어어얼'''
    result = []
    for i in range(len(x)):
        z = []
        for j in range(len(x[i])):
            z.append(x[i][j] + y[i][j])
        result.append(z)
    return result


# #2 subtract

def subtract(x, y):
    ''' 행렬끼리 빼주는 함수'''
    result = []
    for i in range(len(x)):
        z = []
        for j in range(len(x[i])):
            z.append(x[i][j] - y[i][j])
        result.append(z)
    return result



# #3 multiply

def multiply(x, y):
    result = []
    for i in range(len(x)):
        z = []
        for j in range(len(x[i])):
            z.append(x[i][j] * y[i][j])
        result.append(z)
    return result

# #4 divide
def divide(x, y):
    result = []
    for i in range(len(x)):
        z = []
        for j in range(len(x[i])):
            z.append(x[i][j] / y[i][j])
        result.append(z)
    return result


 #5 dot
def dot(x, y):
    ''' 두 행렬 x, y의 dot 연산 결과를 리턴 걍 넘파이를 쓰자
    dot_ik = sum(j)[a_ij * b_jk]

    '''

    # 반복문 여러개 연습

    # 선생님 답
    print('A shape:', A.shape)
    print('B shape:', B.shape)
    if A.shape[1] != B.shape[0]:
        raise ValueError('A의 column과 B의 row 개수는 같아야함')

    n_row = A.shape[0] # dot 결과 행렬의 row 개수
    n_col = B.shape[1] # dot 결과 행렬의 column 개수
    temp = A.shape[1] # 각 원소들끼리 곱한 결과를 더하는 회수
    # numbers = [] # 결과값들을 저장할 리스트 // 비어있는 리스트를 만들지말고
    result = np.zeros((n_row, n_col))
    for i in range(n_row): # A행렬의 row개수만큼 반복
        for k in range(n_col): # B 행렬의 column 개수만큼 반복
            n = 0
            for j in range(temp):
                # dot 결과 행렬의 [i, k]번쨰 원소의 값을 계산
                n += A[i, j] * B[j, k]
            numbers.append(n) # [i, j]번째 원소를 리스트에 추가
    # 결과를 (n_row, n_col) 모양의 행렬로 변환해서 리턴
    return np.array(numbers). reshape(n_row, n_col)




    # 내답
    # result = [len(y[0]) * [0] for i in range(len(x))]
    # for i in range(len(result)):
    #     for j in range(len(result[i])):
    #         for k in range(len(x[i])):
    #             result[i][j] += x[j][k] * y[k][j]
    # return result


"""
6. 항등 행렬(Indentity matrix): 대각선의 원소는 1이고, 나머지 원소는 0인 정사각행렬 ( -> 단위 행렬이라고도 하는듯?)
    A @ I = I @ A = A를 만족
7. 역행렬(Inverse matrix): A @ A- = A- @ A = I를 만족하는 행렬
8. 전치 행렬(Transpose matrix): 행렬의 row와 column을 서로 바꾼 행렬
"""

#6. 항등 행렬
im = np.eye(3, dtype=int)
print(im)

#7. 역행렬
# print(C)
# print(np.linalg.inv(C))

#8. 전치 행렬
print(C)
print(C.transpose())


if __name__ == '__main__':

    a = [[1, 2], [3, 4]]
    b = [[3, 4], [5, 6]]
    print('a:', a)
    print('b:', b)
    print('add:', add(a, b))
    print('subtract :', subtract(a, b))
    print('multiply :', multiply(a, b))
    print('divide :', divide(a, b))
    print('dot:',dot(a, b))
    print(B @ A) # B.dot(A) #@는 numpy array 에서만 쓸수 있는 dot 연산자

