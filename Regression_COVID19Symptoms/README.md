## 프로그램 명칭(국문)
로지스틱 회귀분석을 이용한 코로나 19 바이러스 감염 예측 시뮬레이터

## 프로그램 명칭(영문)
COVID19 iInfection Prediction Simulator Using Logistic Regression

## 발명자 리스트
김연의, 심서영, 장유진

## 프로그램 전반적인 목적
본 프로그램은 체온 데이터, 기침, 콧물과 같은 증상의 유무, 확진자와의 접촉 유무 등의 데이터를 학습하고 Covid-19 감염 가능성을 예측하기 위해 고안된 프로그램이다. 본 프로그램에서는 target data가 1, 0으로 나타나지기 때문에 (감염 = 1, 비감염 = 0) Logistic Regression을 이용한다. 각 feature들의 회귀계수를 구한 후 가중치로 부과하여 70점 만점의 전염병 감염 위험도 점수를 부여한다.

## 데이터셋
https://www.kaggle.com/takbiralam/covid19-symptoms-dataset




