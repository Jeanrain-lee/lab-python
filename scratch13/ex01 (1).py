import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import load_diabetes

# X, y = load_diabetes(return_X_y=True)
# print(X[:5])
# print(X.shape, y.shape)

datasets = load_diabetes()
# print(datasets)
X = datasets.data
y = datasets.target
print(X.shape, y.shape)
features = datasets.feature_names
print('feature names:', features) # 어떤 변수 이름들이 있뉘
print('x[0]=', X[0])
# 모든 특성(컬럼)들이 평균=0, 표준 편차=1 로 전처리가 되어 있는 데이터 세트.
print('y[0]=', y[0])

#dataframe/ ndarray인지 구분
# 나이나 성별에 실수형데이터가 들어가이씀, 모든 변수들이 이미 표준화가 되어있대우에웩


# 선형 회귀(linear regression)
# y = b + a * x
# y = b0 + b1 * x1 + b2 * x2 + ...

# 1개의 figure에 10개의 subplot를 그려서, 변수들과 당뇨병(y)의 대략적이 관계를 파악.
#
# y ~ age, y ~ sex, y ~ bmi, ...


fig, ax = plt.subplots(3, 4)
for row in range(3):
    for col in range(4):
        axis = ax[row, col]
        idx = 4 * row + col
        if idx > 9:
            break
        x = X[:, idx]
        axis.scatter(x, y)
        axis.set_title(features[idx])
plt.show()







# bmI와 당뇨병 수치간 관계의 선형식을 찾아보으아
# y = b + a * bmi -> y와 bmi간의 선형관계식
bmi = X[:, np.newaxis, 2] #BMI가 2번쨰 항목이니까 // np.newaxis : 컬럼으로 하겠다
# scikit.learn의 LinearRegression은 2차원 배열 형태의 훈련 데이트 세트만 사용하기 때문에
print('bmi.shape:', bmi.shape) #1차원 배열
print('bmi[5] =', bmi[:5])

# bmi를 학습(training) / 검증(test) 세트로 분리
bmi_train = bmi[:-40]
bmi_test = bmi[-40:]
y_train = y[:-40]
y_test = y[-40:]

# 선형 회귀 모델(linear regression model) 객체 생성
regr = linear_model.LinearRegression()      #가장 간단한 선형회귀 모델

# training set를 학습(fit) 시킴
# fit만 시키면 y절편과 기울기 결정이 됨
# y = b + a * bmi 선형 관계식에서 y절편 b와 기울기 a를 결정하는게 => 학습시킨다는 과정
regr.fit(bmi_train, y_train)
print('coefficients:', regr.coef_)
# 검증 세트로 테스트
y_pred = regr.predict(bmi_test) # 예측한 예측값

plt.scatter(bmi_test, y_test) # 실제 값
# plt.scatter(bmi_test, y_pred) # 예측값
plt.plot(bmi_test, y_pred, 'ro-') # 예측값
plt.title('Diabetes vs BMI')
plt.show()



# y ~s5 선형 관계식을 찾고, 그래프를 그림

s5 = X[:, np.newaxis, 8] #2차 행렬의 값을 갖기 위해 아무 값도 갖지않은 컬럼을 걍 넣어버렸다는 뜻
print('s5.shape:', s5.shape)

s5_train = s5[:-40]
s5_test = s5[-40:]
y_train = y[:-40]
y_test = y[-40:]

regr = linear_model.LinearRegression()

regr.fit(s5_train, y_train)
print('Coefficient:', regr.coef_) # 선형 회귀식에서 기울기
y_pred = regr.predict(s5_test)
plt.scatter(s5_test, y_test) #실제 값
#plt.scatter(s5_test, y_pred)
plt.plot(s5_test, y_pred, 'yo-')
plt.title('Diabetes vs S5')
plt.xlabel('s5')
plt.show()



array = np.array([[1, 2],
                  [3, 4]])
print(array) # 2 x 2 행렬(2차원 배열)
for row in range(2):
    for col in range(2):
        print(array[row, col], end=' ')
print()


# saint lee 방법쓰
array_flatten = array.flatten() # flatten은 1차원으로 평평하게 만들어버리는
print(array_flatten)
for i in range(4):
    print(array_flatten[i], end=' ')
print()


fig, ax = plt.subplots(3, 4)
# ax : 3 x 4 형태의 2차원 배열(ndarray)
ax_flat = ax.flatten()
for i in range(len(features)):
    subplot = ax_flat[i]
    subplot.scatter(X[:, i], y)
    subplot.set_title(features[i])
plt.show()



# fig, ax = plt.subplots(3, 4)
# for row in range(3):
#     for col in range(4):
#         axis = ax[row, col]
#         idx = 4 * row + col
#         if idx > 9:
#             break
#         x = X[:, idx]
#         axis.scatter(x, y)
#         axis.set_title(features[idx])
# plt.show()



# print(ax) # -> 배열 안에 배열