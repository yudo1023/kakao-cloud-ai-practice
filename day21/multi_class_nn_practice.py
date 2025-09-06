# 1. 목표 : 다중 클래스 분류 신경망 구현
# 2. 요구사항
# 1) 입력층: 64개(8x8 이미지)
# 2) 은닉층: 적절한 크기
# 3) 출력층: 10개(0-9 숫자)
# 4) Softmax 활성화 함수
# 5) Cross-Entropy 손실 함수

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rc('font', family='Apple SD Gothic Neo')
plt.rcParams['axes.unicode_minus'] = False

def mini_mnist():
    """
    8x8 숫자 이미지 분류 (sklearn digits 데이터셋 사용)
    """
    # 데이터 로드
    digits = load_digits()
    X, y = digits.data, digits.target
    X = X / 16.0 # 정규화 (0-16 -> 0-1)

    # 원-핫 인코딩
    y_onehot = np.eye(10)[y]
    num_classes = y_onehot.shape[1]

    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    print("❓ 미니 MNIST 과제")
    print(f"훈련 데이터 : {X_train.shape}")
    print(f"테스트 데이터 : {X_test.shape}")
    print("클래스 수 : 10개 (0-9 숫자)")

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(z):
        s = sigmoid(z)
        return s * (1 - s)
    
    class DigitClassifier:
        def __init__(self, input_size, hidden_size, output_size, lr=0.1):
            # 네트워크 구조 설계
            # input_size = 64, hidden_size = 적절하게 받아오기, output_size = 10
            self.W1 = np.random.randn(input_size, hidden_size) * 0.1
            print(f"\nW1 형태: {self.W1.shape}")
            self.b1 = np.zeros((1, hidden_size))
            # print(f"b1 형태: {self.b1.shape}")
            self.W2 = np.random.randn(hidden_size,output_size) * 0.1
            print(f"W2 형태: {self.W2.shape}")
            self.b2 = np.zeros((1, output_size))
            # print(f"b2 형태: {self.b2.shape}")
            self.lr = lr

            self.z1 = None
            self.a1 = None
            self.z2 = None
            self.a2 = None

        def softmax(self, z):
            # Softmax 함수 구현
            # 벡터를 받아서 그 값을 확률 분포로 변환
            # 출력층을 확률로 바꿔주는 함수, 다중 클래스 분류에 필수
            z = z - np.max(z, axis=1, keepdims=True) # axis=1: 수치안정화 / keepdims=True: (N,1)형태로 유지
            exp_z = np.exp(z)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True) # 각 행의 합이 1이 되도록 나눠주기(확률 분포)

        def cross_entropy_loss(self, y_pred, y_true):
            # Cross-Entropy 손실 구현
            # -np.sum(y_true * np.log(y_pred)
            batch_size = y_pred.shape[0]
            return -np.sum(y_true * np.log(y_pred)) / batch_size
        
        def forward(self, X):
            # 순전파
            self.z1 = np.dot(X, self.W1) + self.b1 # 선형 변환
            # print(f"1단계 - 은닉층 입력 z1 형태 : {self.z1.shape}")
            self.a1 = sigmoid(self.z1) # 활성화 함수
            # print(f"1단계 - 은닉층 출력 a1 형태 : {self.a1.shape}")
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            # print(f"2단계 - 출력층 입력 z2 형태 : {self.z2.shape}")
            self.a2 = self.softmax(self.z2)
            # print(f"2단계 - 최종 출력 a2 형태 : {self.a2.shape}")

            return self.a2

        def backward(self, X, y):
            # 역전파
            N = X.shape[0]

            # 출력층: softmax + cross-entropy -> dz2 = (y_pred - y_true)/N
            dz2 = (self.a2 - y) / N
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            # 은닉층: a1 = sigmoid(z1)
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * sigmoid_derivative(self.z1)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        def predict(self, X):
            # 예측 함수
            z1 = np.dot(X, self.W1) + self.b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            y_hat = self.softmax(z2)
            pred_idx = np.argmax(y_hat, axis=1)

            return pred_idx, y_hat

        def accuracy(self, X, y):
            # 정확도 계산
            pred_idx, _ = self.predict(X)
            true_idx = np.argmax(y, axis=1)

            return (pred_idx == true_idx).mean()

    model = DigitClassifier(input_size=64, hidden_size=256, output_size=num_classes, lr=0.1)
    epochs = 300
    
    losses = []
    for epoch in range(1, epochs + 1):
        output = model.forward(X_train)
        loss = model.cross_entropy_loss(output, y_train)
        model.backward(X_train, y_train)
        losses.append(float(loss))

        train_result = model.accuracy(X_train, y_train)
        test_result = model.accuracy(X_test, y_test)
        print(f"[{epoch:3d}/{epochs}] loss={loss:.4f} | train={train_result*100:.2f}% | test={test_result*100:.2f}%")

    print("\n 🌟 구현 후 다음을 확인하세요:")
    print("- 훈련 정확도: 90% 이상")
    print("- 테스트 정확도: 85% 이상")
    print("- 각 숫자별 예측 성능")

    print(f"\n최종 결과: train={train_result*100:.2f}% | test={test_result*100:.2f}%")

    y_true = np.argmax(y_test, axis=1)
    y_pred, _ = model.predict(X_test)
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(per_class_accuracy):
        print(f"클래스 {i} 정확도: {acc*100:.2f}%")

    if train_result >= 0.90 and test_result >= 0.85:
        print("\n성공!!!!!")
    else:
        print("\n다시 해보자ㅜㅜ~~~~")

    plt.figure(figsize=(12, 9))

    # 손실 그래프
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title('학습 손실 변화 (Cross-Entropy)')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.grid(True)

    # 가중치 변화 시각화(랜덤 200개)
    plt.subplot(2, 2, 2)
    # plt.bar(range(len(model.W1.flatten())), model.W1.flatten()) # 샘플 수가 너무 많아서 그래프가 제대로 표현 안됨
    idx = np.random.choice(model.W1.flatten().size, 200, replace=False) # 랜덤으로 200개정도 뽑아서 표현
    plt.bar(range(200), model.W1.flatten()[idx])
    plt.title('은닉층 가중치 (학습 후)')
    plt.xlabel('가중치 인덱스')
    plt.ylabel('가중치 값')

    # 예측 결과 비교(랜덤 1개) 
    plt.subplot(2, 2, 3)
    idx = np.random.randint(0, X_test.shape[0])
    _, prob_one = model.predict(X_test[idx:idx+1])
    true_one = y_test[idx]
    x_pos = np.arange(10)
    plt.bar(x_pos - 0.2, true_one.flatten(), 0.4, label='정답(원-핫)', alpha=0.7)
    plt.bar(x_pos + 0.2, prob_one.flatten(), 0.4, label='예측 확률', alpha=0.7) 
    plt.title(f'예측 결과 비교 (샘플 idx={idx}, 실제={np.argmax(true_one)}, 예측={np.argmax(prob_one)})')
    plt.xlabel('클래스(0~9)')
    plt.ylabel('값')
    plt.legend()
    plt.xticks(x_pos, list(range(10)))
    
    # 신경망 구조 시각화
    plt.subplot(2, 2, 4)
    plt.text(0.15, 0.7, '입력층\n(64개)', ha='center', va='center',
             bbox=dict(boxstyle="round", facecolor='lightblue'))
    plt.text(0.50, 0.7, f'은닉층\n({model.W1.shape[1]}개)', ha='center', va='center',
             bbox=dict(boxstyle="round", facecolor='lightgreen'))
    plt.text(0.85, 0.7, '출력층\n(10개)', ha='center', va='center',
             bbox=dict(boxstyle="round", facecolor='lightcoral'))

    # 화살표
    plt.arrow(0.22, 0.70, 0.20, 0.0, head_width=0.02, head_length=0.02, fc='black', length_includes_head=True)
    plt.arrow(0.57, 0.70, 0.20, 0.0, head_width=0.02, head_length=0.02, fc='black', length_includes_head=True)

    plt.xlim(0, 1)
    plt.ylim(0.5, 0.9)
    plt.title('신경망 구조')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

mini_mnist()