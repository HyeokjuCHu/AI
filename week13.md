# ECE30008 Intro to AI - Deep Learning 요약

## 1. Deep Learning 개요

### Deep Learning 아키텍처
- **Deep Neural Networks (DNN)**
  - MLP, SOM, RBF, RBM, DBN, DBM 등
- **Convolutional Neural Networks (CNN)**
  - 이미지 데이터에 특화
  - 다양한 레이어 구조 (convolution, pooling 등)
  - 위치 독립적인 지역 특징 학습
- **Recurrent Neural Networks (RNN)**
  - 시계열 데이터에 특화
  - 순환 연결을 통한 메모리 기능
  - 입력 + 컨텍스트 처리

### 모델 발전 역사
- **Building blocks (1986년부터)**: RBM, Auto-encoder, Sparse coding
- **Deep networks (2006년부터)**:
  - **생성 모델**: DBN, GSN, DBM, VAE, GAN, Diffusion model
  - **판별 모델**: CNN, RNN, 생성 모델의 fine tuning

## 2. Artificial Neural Network 기초

### 핵심 개념
- 생물학적 신경망에서 영감을 받은 수학적 모델
- **연결 가중치**에서 지능이 나옴
- 가중치는 데이터로부터 학습됨
- 각 레이어는 입력 정보를 결합하여 상위 레벨 정보 생성
- **특징 추출/추상화**는 내적(dot product)을 통해 수행

### Perceptron의 한계
- 단일 퍼셉트론은 XOR 문제 해결 불가 (비선형 문제)
- **다층 네트워크**로 비선형 문제 해결 가능

### NN vs Deep Network
- **기본 NN**: 1개의 은닉층
- **Deep Network**: 일반적으로 3개 이상의 은닉층

## 3. 활성화 함수 (Activation Functions)

### 필요성
- **비선형성** 제공
- 출력을 특정 범위로 제한
- 측정값을 확률이나 결정으로 변환

### 주요 활성화 함수
- **Sigmoid**: \( \sigma(x) = \frac{1}{1+e^{-x}} \)
- **Tanh**: 하이퍼볼릭 탄젠트
- **ReLU**: \( f(x) = \max(0,x) \)
- **Leaky ReLU**: \( f(x) = \max(ax,x) \)

### 문제점
- **Gradient Vanishing**: 특히 Sigmoid, Tanh에서 발생

## 4. 학습 과정 (Training)

### 학습 절차
1. 훈련 샘플 X와 레이블 c가 주어짐
2. 원하는 출력 D = (d₁, ..., dᵢ, ..., dₙ) 설정
   - dᵢ = 1 (i = c인 경우), 0 (그 외)
3. 손실 함수로 손실 E(W) 계산
4. 경사 기반 알고리즘으로 최적 가중치 W* 찾기: \( W^* = \arg\min_W E(W) \)

### Backpropagation
- 심층 신경망 훈련의 핵심 알고리즘
- 손실 함수의 **그래디언트를 효율적으로 계산**
- 경사 하강법을 통한 매개변수 최적화 가능

## 5. Convolutional Neural Networks (CNNs)

### CNN 개요
- **필터(커널) 최적화**를 통해 특징을 학습하는 피드포워드 신경망
- 이미지 인식, 음성 인식 등 다양한 실용적 응용
- 순수 지도학습으로 훈련 가능한 몇 안 되는 모델 중 하나

### 생물학적 배경
- **Hubel-Wiesel 시스템** (1950년대)
- 시각 피질의 수용장(receptive field) 개념
- **Simple cells**: 지역 특징 감지
- **Complex cells**: Simple cell들의 출력을 풀링

### 역사적 발전
- **Neocognitron** (Fukushima, 1980): S-layer(convolution), C-layer(downsampling) 도입
- **CNN** (LeCun, 1989): 현대적 CNN의 기초

### CNN 구조 요소

#### 1. Convolution 연산
- **커널(필터)**: 특징 감지기 역할
- **스트라이드**: 필터 이동 간격
- **패딩**: 입력 크기 조정

#### 2. Pooling
- **목적**: 고해상도가 불필요한 경우 계산 비용 절약
- **Max Pooling**: 영역 내 최댓값 선택
- **Average Pooling**: 영역 내 평균값 계산

#### 3. 활성화 함수
- 주로 **ReLU** 사용
- Gradient vanishing 문제 완화

### CNN 레이어 처리 과정
1. **Convolution**: 필터 적용으로 특징맵 생성
2. **Activation**: ReLU 등 활성화 함수 적용
3. **Pooling**: 다운샘플링으로 크기 축소
4. **반복**: 여러 레이어를 통해 점진적 특징 추출
5. **Fully Connected**: 최종 분류를 위한 완전연결층

### 주요 CNN 아키텍처

#### LeNet (1989)
- 최초의 실용적 CNN
- 손글씨 숫자 인식에 사용

#### AlexNet (2012)
- ImageNet 대회에서 혁신적 성과
- 8개 레이어, 60M 매개변수
- GPU 활용한 훈련

#### VGG Network
- 단순하고 인기 있는 구조
- 3×3 필터 일관 사용
- 깊이 증가로 성능 향상

#### GoogleNet
- 22개 레이어
- Inception 모듈 도입
- 효율적인 계산 구조

### ImageNet Challenge 성과
- 2012년 AlexNet: Top-5 에러율 17.0%
- 지속적인 성능 향상으로 인간 수준 달성
- 152개 레이어 ResNet: 3.57% 에러율

## 6. 손실 함수와 최적화

### Softmax 함수
신경망 출력을 확률로 변환:
$$\text{softmax}(y)_i = \frac{\exp(y_i)}{\sum_j \exp(y_j)}$$

### Cross-entropy 손실
$$H(p,q) = -\sum_x p(x) \log q(x)$$
- p: 실제 분포 (ground truth)
- q: 예측 분포 (softmax 출력)

## 7. PyTorch 프레임워크

### 주요 특징
- Python 기반 과학 계산 패키지
- GPU 활용으로 Numpy 대비 성능 향상
- 딥러닝을 위한 최대 유연성과 속도 제공

### 핵심 구성요소

#### Autograd
- 모든 텐서 연산에 대한 **자동 미분** 제공
- `.backward()` 호출로 그래디언트 자동 계산
- 순전파/역전파 자동화

#### torch.nn
- 신경망 모듈 제공
- Linear, CNN, RNN 등 쉬운 구현
- 모듈화된 네트워크 구조 설계

#### 손실 함수와 최적화기
- **손실 함수**: MSE, CrossEntropy, L1, NLL 등
- **최적화기**: SGD, Adam, RMSProp, AdaDelta 등

### 텐서 연산
- Torch 텐서 ↔ Numpy 배열 간 쉬운 변환
- CUDA 텐서로 GPU 활용 가능
- `.to()` 메서드로 디바이스 간 이동

이러한 딥러닝 기술들은 현재 컴퓨터 비전, 자연어 처리, 음성 인식 등 다양한 AI 분야에서 핵심 기술로 활용되고 있습니다.