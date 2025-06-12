# ECE30008 Introduction to AI – Machine Learning

## 1. 머신러닝의 정의
- **Tom Mitchell (1997)**  
  “A computer program is said to **learn** from experience **E** with respect to some class of tasks **T** and performance measure **P**, if its performance at the tasks improves with the experiences.”
- **핵심 질문**: How to build programs that improve with experience?
- **프로세스**:  
  ```
  초기 모델 + 데이터 + 목적함수 → 최적화 → 훈련된 모델
  ```

## 2. ML 워크플로우
1. 데이터 수집 (Acquisition)  
2. 데이터 준비 (Preparation)  
3. 분석 (Analysis)  
4. 모델링 (Modeling)  
5. 시각화 (Visualization)  
6. 배포 & 유지보수 (Deployment & Maintenance)

## 3. ML의 4가지 구성 요소
- **데이터**: 특징, 라벨, 훈련/검증/테스트  
- **모델**: SVM, 신경망, 나이브 베이즈, 로지스틱 회귀, 랜덤 포레스트 등  
- **목적함수**: 교차엔트로피, RMSE, 우도 등  
- **최적화**: 경사하강법, 뉴턴법, 선형/볼록 최적화

## 4. 머신러닝 카테고리
- **지도학습**  
  - 분류 (Classification)  
  - 회귀 (Regression)  
- **비지도학습**: 클러스터링, 차원 축소  
- **준지도학습**: 라벨된 + 라벨되지 않은 데이터  
- **강화학습**  
  - Credit Assignment  
  - Exploration vs Exploitation

## 5. 판별 모델 vs 생성 모델
- **판별 모델**  
  - \(p(t \mid x)\)  
  - 결정 경계 학습, 지도학습 전용  
- **생성 모델**  
  - \(p(t,x)\) 또는 \(p(x \mid t)\)  
  - 데이터 분포 모델링, 반지도/비지도 활용

## 6. 데이터 표현
- \(X \in \mathbb{R}^{N\times D}\): N개 샘플, D개 특징  
- \(Y \in \mathbb{R}^N\): N개 라벨

## 7. 훈련 / 검증 / 테스트
- **훈련**: 매개변수 학습  
- **검증**: 하이퍼파라미터 선택, 과적합 방지  
- **테스트**: 최종 성능 평가  
> 테스트 데이터는 절대 훈련에 사용 금지

## 8. 교차 검증
1. 랜덤 서브샘플링  
2. K-fold 교차 검증  
3. Leave-one-out 교차 검증

## 9. 과적합과 방지
- **과적합**: 훈련 데이터에 과도하게 적합  
- **방지 방법**  
  1. 더 많은 데이터  
  2. 더 단순한 모델  
  3. 정규화  
  4. 조기 중단 (Early Stopping)  
- **Occam’s Razor**: 불필요한 복잡성은 제거

## 10. Scikit-Learn 핵심 클래스
1. `sklearn.datasets`  
2. `sklearn.tree`, `sklearn.svm`, `sklearn.neighbors` 등  
3. `sklearn.metrics`  
4. `sklearn.model_selection`

## 11. 지도학습 워크플로우
1. 데이터/라벨 분할  
2. 하이퍼파라미터 튜닝  
3. 모델 훈련  
4. 모델 테스트  
5. 최종 모델 학습

## 12. 딥러닝과 AI 발전
- 복잡한 문제 해결을 위한 강력한 모델  
- AI 발전: 초기 AI → 머신러닝 → 딥러닝

## 13. 기말고사 핵심 암기 사항
1. Tom Mitchell 정의 (E, T, P)  
2. ML 워크플로우 6단계  
3. ML 4대 구성 요소  
4. 4가지 학습 카테고리  
5. 판별 vs 생성 모델 차이  
6. 과적합 방지 4가지 방법  
7. 교차 검증 3가지 방법  
8. Scikit-Learn 4대 클래스  
9. 지도학습 5단계  
10. 테스트 데이터 독립성