import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(10000)

#데이터 읽기
symptom = pd.read_csv("covid-19 symptoms dataset.csv")
print(symptom.head())

#사용 변수 지정
feature_columns = symptom.columns.difference(["infectionProb"])
X = symptom[feature_columns]
y = symptom["infectionProb"]

#training set/ test set 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size=0.7, test_size=0.3, random_state = 123)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#로지스틱회귀분석
model = sm.Logit(y_train, X_train)
results = model.fit(model = "newton")

#회귀계수 확인
cof = np.exp(results.params)
print(cof)


y_pred = results.predict(X_test)

#인계값 함수
def cut_off(y, threshold):
    Y = y.copy()
    Y[Y > threshold] = 1
    Y[Y < threshold] = 0
    return Y.astype(int)


Y_pred = cut_off(y_pred, 0.478)


#ACC값 구하는 함수
def acc(cfmat):
    return (cfmat[0, 0] + cfmat[1, 1]) / (cfmat[0, 0] + cfmat[1, 1] + cfmat[0, 1] + cfmat[1, 0])


cfmat = confusion_matrix(y_test, Y_pred)


#confusion matrix
import seaborn as sns
class_names=[0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cfmat), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


#AUC 구하기
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
auc(fpr, tpr)


#RMSE 구하기
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, Y_pred))

size = len(feature_columns)

#가중치 구하기
def Weight(siz, cof):
    sum = 0
    for i in range (size):
        sum += float(cof[i])

    coe = []
    for i in range (size):
        coe.insert(i, float(cof[i])/sum*70)

    return coe

coe = Weight(size, cof)


#입력 받기
def Input(size):
    Condition = []
    print("질문에 답하세요")
    Condition.append(float((input("체온이 얼마인가요?(C) : "))))
    Condition.append(int((input("몸이 아픈가요?(네 = 1, 아니요 = 0) : "))))
    Condition.append(int((input("나이가 몇 살인가요? : "))))
    Condition.append(int((input("콧물이 나오나요??(네 = 1, 아니요 = 0) : "))))
    Condition.append(int((input("숨쉬는게 힘든가요?(아프다 = 1, 보통 = 0, 안아프다 = -1) : "))))
    Condition.append(int((input("피곤한가요?(네 = 1, 아니요 = 0) : "))))
    Condition.append(int((input("마른기침이 나오나요??(네 = 1, 아니요 = 0) : "))))
    Condition.append(int((input("목이 아픈가요?(네 = 1, 아니요 = 0) : "))))
    Condition.append(int((input("Covid-19에 걸렸던적이 있나요?(네 = 1, 아니요 = 0) : "))))
    Condition.append(int((input("코가 막혔나요?(네 = 1, 아니요 = 0) : "))))
    Condition.append(int((input("설사를 했나요?(네 = 1, 아니요 = 0) : "))))
    contact = input("확진자와 접촉했나요?(네 = 1, 아니요 = 0, 모릅니다 = -1): ")
    if contact == 1:
        Condition.append(0)
        Condition.append(0)
        Condition.append(1)
    elif contact == 0:
        Condition.append(0)
        Condition.append(1)
        Condition.append(0)
    else :
        Condition.append(1)
        Condition.append(0)
        Condition.append(0)
    return Condition

Condition = Input(size)

def Calculate(size, coe, Condition):
    score = 0
    if Condition[0] > 37.5:
        Condition[0] = 1
    else :
        Condition[0] = 0

    if Condition[2] > 60 or coe[2] < 15:
        Condition[2] = 1
    else:
        Condition[2] = 0

    for i in range(size):
        weight = coe[i]
        satus = Condition[i]
        score = score + weight * satus
    return score

print(Calculate(size, coe, Condition))
