from SpO2HR.Score import HPSPO2score
from Regression_COVID19Symptoms.regression import COVID_SymptomsScore

Underlying_disease = int(input("기저질환을 가지고 계신가요?(네 = 1, 아니요 = 0) : "))

Final_Score = HPSPO2score+COVID_SymptomsScore
if Underlying_disease == 0:
    Final_Score += 0
else:
    Final_Score += 0

print("코로나19 감염 위험도: %.2f " % (Final_Score))
