# Classification 요약

## 1. 학습 전략 복습

### 머신러닝 분류
- **비지도 학습(Unsupervised Learning)**: 클러스터링
  - 데이터를 "자연스러운" 분할로 조직화하는 문제
- **지도 학습(Supervised Learning)**:
  - **분류(Classification)**: yes/no, True/False, 0/1과 같은 범주형 해답을 가진 문제
  - **회귀(Regression)**: "제품 가격"이나 "수익"과 같은 연속값 예측 문제

## 2. 분류 알고리즘들

### 주요 분류 알고리즘
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
  - 베이즈 정리 기반: \( P(A|B) = \frac{P(B|A)P(A)}{P(B)} \)
  - 사후확률 = \( \frac{\text{사전확률} \times \text{우도}}{\text{증거}} \)
- **K-Nearest Neighbors (KNN)**
- **Decision Tree / Random Forest**

## 3. Decision Tree (의사결정트리)

### 기본 개념
- **정의**: 행동 과정을 나타내는 트리 형태의 예측 모델
- **구조**: 각 가지는 가능한 결정이나 행동을 나타냄
- **유형**:
  - **분류 트리**: True/False, Yes/No와 같은 범주형 문제
  - **회귀 트리**: 연속값 예측 문제

### 트리 구조 용어
- **Root Node**: 초기 질문
- **Interior(Internal) Nodes**: 후속 질문들
- **Leaf Nodes**: 결정 노드, 예측 수행

### 핵심 질문들
- **예측**: 주어진 의사결정트리로 어떻게 예측을 수행하는가?
- **학습**: 데이터로부터 의사결정트리를 어떻게 훈련시키는가?
  - 어떤 질문을 언제 할 것인가?

## 4. Decision Tree 학습 과정

### 예측 과정
1. 훈련된 의사결정트리가 주어졌다고 가정
2. 각 데이터 인스턴스에 대해 각 노드의 질문에 답하고 해당 가지를 선택
3. 리프 노드에 도달하면 해당 노드의 결정에 따라 예측 수행
4. 재귀 프로그램으로 효율적 구현 가능

### 훈련 과정
1. 빈 의사결정트리에서 시작, 모든 훈련 데이터 포함
2. 다음 최적 속성(특성)으로 분할
3. 정보 이득(Information Gain)을 사용하여 분할 조건 결정

## 5. 분할 기준 - 정보 이득

### 엔트로피(Entropy)
- **정의**: 데이터의 복잡도 측정
- **수식**: \( H(Y) = -\sum_i p(y_i) \log_2 p(y_i) \)
- **높은 엔트로피**: 균등 분포, 예측하기 어려움
- **낮은 엔트로피**: 편향된 분포, 예측하기 쉬움

### 정보 이득(Information Gain)
- **정의**: 노드 분할 후 엔트로피(불확실성) 감소량
- **수식**: \( IG(X) = H(Y) - H(Y|X) \)
- **조건부 엔트로피**: \( H(Y|X) = \sum_j p(x_j) \left(-\sum_i p(y_i|x_j) \log_2 p(y_i|x_j)\right) \)

### Gini Index (대안적 분할 기준)
- **수식**: \( Gini = \sum_i p(y_i)(1-p(y_i)) = 1 - \sum_i p(y_i)^2 \)
- **의미**: 무작위로 선택된 인스턴스가 잘못 분류될 확률
- **범위**: 0(최소) ~ 1(최대), 낮을수록 좋음

## 6. Decision Tree의 장단점

### 장점
- 이해하고 해석하기 쉬움
- 데이터 전처리 노력이 적음 (정규화 불필요)
- 비선형 결정 경계 획득 가능
- 수치형과 범주형 데이터 모두 처리 가능

### 단점
- **과적합**: 알고리즘이 노이즈를 학습하는 경우
- **불안정성**: 데이터의 작은 변화에도 모델이 크게 변할 수 있음

## 7. Random Forest

### 배경과 해결책
- **문제**: Decision Tree의 과적합과 불안정성
- **해결책**: 
  - 조기 중단 (고정 깊이, 최소 리프 노드 개수)
  - 가지치기 (Pruning)

### Random Forest 개념
- **정의**: 의사결정트리들의 앙상블(혼합)
- **목적**: 
  - 예측 정확도 향상
  - 과적합 방지
  - 안정성과 정확도 개선
- **원리**: 약한 분류기들을 결합하여 강한 분류기 생성

### 다양성 확보 방법
- **배깅(Bagging)**: Bootstrap Aggregating
  - **Bootstrap**: 복원 추출을 통한 무작위 재샘플링
  - 원본 데이터의 다른 부분집합으로 의사결정트리 학습
- **그룹 결정**: 각 트리의 결정을 다수결 투표로 집계

## 8. Scikit-learn 구현

### 기본 사용법
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest  
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 성능 평가
confusion_matrix(y_test, dt_pred)
confusion_matrix(y_test, rf_pred)
```

### 시각화
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix 히트맵
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.ylabel('Predicted label')
plt.xlabel('True Label')
```

## 9. 핵심 개념 정리

- **Information Gain**: 최적 분할 기준 선택
- **Entropy vs Gini**: 두 가지 불순도 측정 방법
- **Overfitting**: Decision Tree의 주요 문제점
- **Ensemble Method**: Random Forest를 통한 성능 개선
- **Bootstrap Sampling**: 다양성 확보를 위한 핵심 기법
- **Majority Voting**: 여러 모델의 결정을 통합하는 방법