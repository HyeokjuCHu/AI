# Linear Regression 요약

## 1. 기본 개념

### Linear Regression이란?
- 두 변수 간의 관계를 선형 방정식으로 모델링하는 기법
- 수식: \( \hat{y}(w,x) = w_0 + w_1x_1 + ... + w_px_p \)
- \( w = (w_1, ..., w_p) \): 계수(coef_), \( w_0 \): 절편(intercept_)

### 주요 용어
- **X (독립변수)**: 예측에 사용되는 변수 (predictor, explanatory variable)
- **Y (종속변수)**: 예측하고자 하는 변수 (response)
- **관계식**: \( Y = f(X) \)

## 2. 학습 과정

### Gradient Descent 방법
- **목적**: 손실함수를 최소화하는 최적의 가중치 찾기
- **손실함수**: \( J(w) = (wx - y)^2 \)
- **가중치 업데이트**: \( w_{i+1} = w_i - \alpha_i[2x(w_ix - y)] \)
- **학습률(α)**: 가중치 업데이트 크기를 조절하는 하이퍼파라미터

### 핵심 용어들
- **Model**: 훈련 데이터로부터 학습한 시스템의 표현
- **Bias**: 원점으로부터의 절편 또는 오프셋
- **Learning Rate**: 경사하강법에서 사용되는 스칼라 값
- **Epoch**: 전체 데이터셋을 한 번 완전히 훈련하는 과정
- **Loss**: 모델 예측값과 실제값 간의 차이
- **Weight**: 선형 모델에서 특성에 대한 계수

## 3. 성능 평가

### R² Score
- **수식**: \( R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{j=1}^N (y_j-\hat{y}_j)^2}{\sum_{j=1}^N (y_j-\bar{y})^2} \)
- **의미**: 모델이 데이터의 분산을 얼마나 잘 설명하는지 측정
- **최적값**: R² = 1 (완벽한 예측)

## 4. 확장된 모델들

### Quadratic Regression
- **목적**: 직선으로 설명되지 않는 비선형 관계 모델링
- **수식**: \( y = b + w_1x + w_2x^2 \)
- **방법**: 원래 특성에 제곱 특성을 추가하여 선형회귀 적용

### Multiple Regression
- **정의**: 여러 독립변수를 사용한 회귀
- **수식**: \( \hat{y}(w,x) = w_0 + w_1x_1 + ... + w_px_p \)

## 5. 과적합 문제와 해결책

### Overfitting vs Underfitting
- **Overfitting**: 모델이 훈련 데이터의 세부사항과 노이즈까지 학습하여 새로운 데이터에 대한 성능이 떨어지는 현상
- **Underfitting**: 모델이 훈련 데이터도 제대로 학습하지 못하는 현상

### Regularization
- **목적**: 모델의 복잡도를 줄여 과적합 방지
- **원리**: 복잡하거나 유연한 모델 학습을 억제

#### L1 Regularization (Lasso)
- **수식**: \( \sum_{i=1}^M (y_i - \sum_{j=0}^p w_j \times x_{ij})^2 + \lambda \sum_{j=0}^p |w_j| \)
- **특징**: 
  - 계수의 절댓값에 페널티 부여
  - 일부 계수를 0으로 만들어 특성 선택 효과
  - 특성 선택과 과적합 방지 동시 달성

#### L2 Regularization (Ridge)
- **수식**: \( \sum_{i=1}^M (y_i - \sum_{j=0}^p w_j \times x_{ij})^2 + \lambda \sum_{j=0}^p w_j^2 \)
- **특징**:
  - 계수의 제곱에 페널티 부여
  - 계수를 0에 가깝게 축소하지만 완전히 0으로 만들지는 않음
  - 모델 복잡도와 다중공선성 감소

## 6. Scikit-learn 구현

### 기본 사용법
```python
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# 기본 선형회귀
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)  # R² 반환

# Lasso 회귀
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Ridge 회귀  
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
```

### 주요 메서드
- **fit()**: 모델 훈련
- **predict()**: 예측 수행
- **score()**: R² 계수 반환
- **coef_**: 학습된 계수
- **intercept_**: 학습된 절편