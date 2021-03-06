import numpy as np

from SpO2HR.prediction import trainPredict, testPredict

Prediction = testPredict  # 웨어러블 기기에서 추출한 데이터로 예측.

AverageHR = np.mean(Prediction[:, 0])
AverageSPO2 = np.mean(Prediction[:, 1])

HRCondition = ['정상', '낮음', '높음']
SPO2Condition = ['정상', '낮음', '매우 낮음']

HPSPO2score = 0

if 70 <= AverageHR <= 80:
    print('Heart Rate: ', HRCondition[0])
    HPSPO2score += 0
elif AverageHR < 70:
    print('Heart Rate: ', HRCondition[1])
    HPSPO2score += 2
elif 80 <= AverageHR < 90:
    print('Heart Rate: ', HRCondition[2])
    HPSPO2score += 5
elif 90 <= AverageHR < 100:
    print('Heart Rate: ', HRCondition[2])
    HPSPO2score += 7
elif AverageHR >= 100:
    print('Heart Rate: ', HRCondition[2])
    HPSPO2score += 10

if AverageSPO2 >= 80:
    print('SPO2: ', SPO2Condition[0])
    HPSPO2score += 0
elif 60 <= AverageSPO2 < 80:
    print('SPO2: ', SPO2Condition[1])
    HPSPO2score += 5
elif 40 <= AverageSPO2 < 60:
    print('SPO2: ', SPO2Condition[1])
    HPSPO2score += 7
elif AverageSPO2 < 40:
    print('SPO2: ', SPO2Condition[2])
    HPSPO2score += 10

print('Heart Rate & SPO2 SCORE: ', HPSPO2score)




