import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
plt.rc('font', family='AppleGothic')

# # ** 딥러닝 기초 **
# # xor
# X = np.array([[0, 0],
#              [0, 1],
#              [1, 0],
#              [1, 1]])

# y = np.array([[0],
#              [1],
#              [1],
#              [0]])

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(z):
#     s = sigmoid(z)
#     return s * (1 - s)

# class SimpleNeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size):
#         self.W1 = np.random.randn(input_size, hidden_size) * 0.5
#         print(f"W1 형태 : {self.W1.shape}")

#         self.b1 = np.zeros((1, hidden_size))
#         print(f"b1 형태 : {self.b1.shape}")

#         self.W2 = np.random.randn(hidden_size, output_size) * 0.5
#         print(f"W2 형태 : {self.W2.shape}")

#         self.b2 = np.zeros((1, output_size))
#         print(f"b2 형태 : {self.b2.shape}")

#         self.z1 = None
#         self.a1 = None
#         self.z2 = None
#         self.a2 = None

#     # 순전파
#     def forward(self, X):
#         self.z1 = np.dot(X, self.W1) + self.b1 # 선형 변환
#         print(f"1단계 - 은닉층 입력 z1 형태 : {self.z1.shape}")
#         self.a1 = sigmoid(self.z1) # 활성화 함수
#         print(f"1단계 - 은닉층 출력 a1 형태 : {self.a1.shape}")

#         self.z2 = np.dot(self.a1, self.W2) + self.b2
#         print(f"2단계 - 출력층 입력 z2 형태 : {self.z2.shape}")
#         self.a2 = sigmoid(self.z2)
#         print(f"2단계 - 최종 출력 a2 형태 : {self.a2.shape}")

#         return self.a2
    
#     # 역전파
#     def backward(self, X, y, Learning_rate):
#         m = X.shape[0]

#         # dL/da2 -> dL/dz2 -> dL/dW2 -> dL/db2 -> dL/da1 -> dL/dz1 -> dL/dW1 -> dL/db1

#         # dL/da2 = (self.a2 - y)
#         # dL/da2 = 2/m * (예측값 - 정답) -> dL/da2 = 예측값 - 정답
#         # 1) 2는 상수라서 신경망이 학습할때 학습률(learning_rate)에 흡수되기 때문에 생략 가능. 속도는 중요하지 않음 방향이 중요
#         # 2) 예측값 - 정답을 먼저 계산한 뒤 가중치 미분(dW) 계산할 때 평균을 내기 때문에 /m 생략 가능
#         # da2/dz2 = self.a2 * (1 - self.a2) 시그모이드 함수의 미분 공식
#         # dL/dz2 = dL/da2 * da2/dz2 = (self.a2 - y) * self.a2 * (1 - self.a2)
#         dl_dz2 = (self.a2 - y) * self.a2 * (1 - self.a2)
        
#         # dL/dW2 = a1i.T @ dz2i = np.dot(self.a1.T, dz2)
#         # 손실은 MSE이므로 여기서 평균을 나눔
#         dl_dW2 = (1/m) * np.dot(self.a1.T, dl_dz2)

#         # db2 = dz2 모든 샘플의 합
#         # MSE 이므로 전체 샘플에 평균
#         dl_db2 = (1/m) * np.sum(dl_dz2, axis=0, keepdims=True)

#         # dL/da1 = dL/dz2 * dz2/da1 = dL/dz2 * W2 = dL_dz2 @ W2.T
#         # dz2/da1 = W2
#         dl_da1 = np.dot(dl_dz2, self.W2.T)

#         # dL/dz1 = dL/da1 * da1/dz1 = dz2 @ w2.T * sigmoid_derivative(z1)
#         dl_dz1 = dl_da1 * sigmoid_derivative(self.z1)
        
#         dl_dW1 = (1/m) * np.dot(X.T, dl_dz1)
#         dl_db1 = (1/m) * np.sum(dl_dz1, axis=0, keepdims=True)

#         self.W2 -= Learning_rate * dl_dW2
#         self.b2 -= Learning_rate * dl_db2
#         self.W1 -= Learning_rate * dl_dW1
#         self.b1 -= Learning_rate * dl_db1

#         return dl_dW1, dl_db1, dl_dW2, dl_db2
    
#     # 학습
#     def train(self, X, y, epochs, Learning_rate=0.1):
#         losses = []
#         for epoch in range(epochs):
#             # 순전파
#             output = self.forward(X)
#             # 손실계산
#             loss = np.mean((output - y)**2)
#             losses.append(loss)
#             # 역전파
#             self.backward(X, y, Learning_rate)

#             if epoch % 100 == 0:
#                 print(f"에포크 {epoch}, 손실 : {loss:.4f}")

#         return losses
    
# nn = SimpleNeuralNetwork(2, 4, 1)
# print("학습 전")
# before_output = nn.forward(X)
# print(f"예측값 : {before_output.flatten()}")
# print(f"정답값 : {y.flatten()}")

# losses = nn.train(X, y, epochs=10000, Learning_rate=1.0)
# print("학습 후")
# after_output = nn.forward(X)
# print(f"예측값 : {after_output.flatten()}")
# print(f"학습값 : {y.flatten()}")

# plt.figure(figsize=(12, 8))

# # 손실 그래프
# plt.subplot(2, 2, 1)
# plt.plot(losses)
# plt.title('학습 손실 변화')
# plt.xlabel('에포크')
# plt.ylabel('손실(MSE)')
# plt.grid(True)

# # 가중치 변화 시각화
# plt.subplot(2, 2, 2)
# plt.bar(range(len(nn.W1.flatten())), nn.W1.flatten())
# plt.title('은닉층 가중치 (학습 후)')
# plt.xlabel('가중치 인덱스')
# plt.ylabel('가중치 값')

# # 예측 결과 비교
# plt. subplot(2, 2, 3)
# x_pos = np.arange(len(y))
# plt.bar(x_pos - 0.2, y.flatten(), 0.4, label='정답', alpha=0.7)
# plt.bar(x_pos + 0.2, after_output.flatten(), 0.4, label='예측', alpha=0.7)
# plt.title('예측 결과 비교')
# plt.xlabel('샘플')
# plt.ylabel('값')
# plt.legend()
# plt.xticks(x_pos, ['[0,0]', '[0,1]', '[1,0]', '[1,1]'])

# # 신경망 구조 시각화
# plt.subplot(2, 2, 4)
# plt.text(0.1, 0.7, '입력층\n(2개)', ha='center', va='center',
#          bbox=dict(boxstyle="round", facecolor='lightblue'))
# plt.text(0.5, 0.7, '은닉층\n(4개)', ha='center', va='center',
#          bbox=dict(boxstyle="round", facecolor='lightgreen'))
# plt.text(0.9, 0.7, '출력층\n(1개)', ha='center', va='center',
#          bbox=dict(boxstyle="round", facecolor='lightcoral'))

# # 화살표 그리기
# plt.arrow(0.2, 0.7, 0.2, 0, head_width=0.02, head_length=0.02, fc='black')
# plt.arrow(0.6, 0.7, 0.2, 0, head_width=0.02, head_length=0.02, fc='black')

# plt.xlim(0, 1)
# plt.ylim(0.5, 0.9)
# plt.title('신경망 구조')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# print("\n학습 완료")
# print(f"최종 손실 : {losses[-1]:.4f}")
# print(f"XOR 문제 해결 정확도 : {np.mean(np.abs(after_output - y) < 0.1) *100:.1f}%")


# cnn
# CIFAR-10 분류
def create_cnn():
    model = models.Sequential([
        # block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model

model = create_cnn()
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 정확도 : {test_acc:.2f}")

indices = np.random.choice(x_test.shape[0], size=5, replace=False)
test_images = x_test[indices]
test_labels = y_test[indices]

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 4))

for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(test_images[i])
    plt.title(f"Ture: {test_labels[i][0]}, Pred: {predicted_labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()