import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

 # p.152 데이터 정규화 및 표준화
np.random.seed(42)
data = {
    'Age': np.random.randint(20, 70, 100),
    'Salary': np.random.normal(50000, 15000, 100),
    'Experience': np.random.exponential(5, 100),
    'Score': np.random.uniform(0, 100, 100)
}
df = pd.DataFrame(data)
df['Salary'] = df['Salary'].clip(lower=20000)
df['Experience'] = df['Experience'].clip(upper=20)
print(df.describe())
# 표준화(StandardScaler)
# 평균 0, 포준편자 1로 변환 : (x-mean)/std
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(
    scaler_standard.fit_transform(df),
    columns=df.columns
) 
print("표준화 후 통계 : ", df_standardized.describe())
# 정규화(MinMaxScaler)
# 0-1 범위로 변환 : (x-min)/(max-min)
scaler_normalized = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler_normalized.fit_transform(df),
    columns=df.columns
)
print("정규화 후 통계 : ", df_normalized.describe())
# RobustScaler
# 중앙값과 IQR 사용 ; (x-median)/IQR
scaler_robust = RobustScaler()
df_robust = pd.DataFrame(
    scaler_robust.fit_transform(df),
    columns=df.columns
)
print("로버스트 스케일링 후 통계 : ", df_robust.describe())
# 예시) 온라인 쇼핑몰 고객 데이터 전처리
n_customers = 1000
customer_data = {
    'customer_id': range(1, n_customers + 1),
    'name': [f'Customer_{i}' for i in range(1, n_customers + 1)],
    'age' : np.random.normal(35, 12, n_customers).astype(int),
    'gender': np.random.choice(['M', 'F', 'Male', 'Female', 'm', 'f', ''], n_customers),
    'city': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', ''], n_customers),
    'total_purchase': np.random.exponential(50000, n_customers),
    'purchase_count': np.random.poisson(5, n_customers),
    'last_purchase_days': np.random.randint(1, 365, n_customers),
    'membership_level': np.random.choice(['Bronze', 'Sliver', 'Gold', 'Platinum', ''], n_customers)
}
df2 = pd.DataFrame(customer_data)
print("=== 1단계 : 원본 데이터 탐색 ===")
print(f"데이터 크기: {df2.shape}")
print("\n데이터 타입 : ", df2.dtypes)
print("\n처음 5행 : ", df2.head())
print("\n=== 2단계 : 원본 데이터 문제 파악 ===")
print("결측치 현황 : ")
missing_data = df2.isnull().sum()
print(missing_data)
missing_percentage = (missing_data / len(df2))*100
missing_summary = pd.DataFrame({
    '결측지_개수': missing_data,
    '결측지_비율(%)': missing_percentage
})
print(missing_summary[missing_summary['결측치_개수']>0])
print(f"\n데이터 타입 문제 : ")
print(f"나이 데이터 타입 : {df2['age'].dtype}")
print(f"나이 범위 : {df2['age'].min()}~{df2['age'].max()}")
print(f"비정상적인 나이 값: {df2[(df2['age'] < 0) | (df2['age'] > 100 )]['age'].tolist()}")
print(f"\n성별 데이터 일관성 문제 : ")
print(f"성별 고유값 : {df2['gender'].unique()}")
print(f"성별 값 개수 : {df2['gender'].value_counts()}")
print(f"\n구매 금액 이상치 확인 : ")
Q1 = df2['total_purchase'].quantile(0.25)
Q3 = df2['total_purchase'].quantile(0.25)
IQR = Q3 - Q1
outlier_threshold_low = Q1 - 1.5 * IQR
outlier_threshold_high = Q3 + 1.5 * IQR
outliers = df2[(df2['total_purchase'] < outlier_threshold_low) |
              (df2['total_purchase'] > outlier_threshold_high)]
print(f"이상치 개수 : {len(outliers)}개 ({len(outliers)/len(df2)*100:.1f}%)")
print(f"이상치 범위 : {outlier_threshold_low:.0f} 미만 또는 {outlier_threshold_high:.0f} 초과")
print(f"\n중복 데이터 확인 : ")
duplicates = df2.duplicated()
print(f"완전 중복 행 : {duplicates.sum()}개")
name_duplicates = df2.duplicated(subset=['name'])
print(f"이름 중복 : {name_duplicates.sum()}개")
print("\n=== 3단계 : 원본 데이터 전처리 ===")
# median
df2_original = df2.copy()
median_age = df2[(df2['age']>=0)&(df2['age']<=100)]['age'].median()
df2.loc[(df2['age']<0)|(df2['age']>100), 'age'] = median_age
print(f"정제 후 나이 범위 : {df2['age'].min()} ~ {df2['age'].max()}")
print(f"나이 중앙값으로 대체 : {median_age}세")
# 성별 데이터 표준화
gender_mapping = {
    'M': 'Male', 'm': 'Male', 'Male': 'Male',
    'F': 'Female', 'f': 'Female', 'Female': 'Female',
    '': 'Unknown'
}
df2['gender'] = df2['gender'].map(gender_mapping).fillna('Unknown')
print(f"표준화 후 성별 값 : {df2['gender'].unique()}")
print(f"성별 분포 : \n{df2['gender'].value_counts()}")
# 도시 데이터 결측치 처리
df2['city'] = df2['city'].replace('', np.nan)
most_common_city = df2['city'].mode()[0]
df2['city'] = df2['city'].fillna(most_common_city)
print(f"결측치 처리 후 도시 분포 : \n{df2['city'].value_counts()}")
print(f"최빈값 '{most_common_city}'로 결측치 대체")
# 멤버십 레벨 결측치 처리
df2['membership_level'] = df2['membership_level'].replace('', 'Bronze')
print(f"멤버십 레벨 분포 : \n{df2['membership_level'].value_counts()}")
# 구매 금액 이상치 처리
Q1 = df2['total_purchase'].quantile(0.25)
Q3 = df2['total_purchase'].quantile(0.75) 
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df2.loc[df2['total_purchase'] < lower_bound, 'total_purchase'] = lower_bound
df2.loc[df2['total_purchase'] > upper_bound, 'total_purchase'] = upper_bound
print(f"이상치 처리 후 구매 금액 통계 : ", df2['total_purchase'].describe())
# 파생 변수 생성
df2['avg_purchase_amount'] = df2['total_purchase'] / df2['purchase_count']
df2['avg_purchase_amount'] = df2['avg_purchase_amount'].fillna(0)
df2['customer_segment'] = 'Regular'
df2.loc[(df2['total_purchase'] > df2['total_purchase'].quantile(0.8)) &
        (df2['last_purchase_days'] < 30), 'customer_segment'] = 'VIP'
df2.loc[(df2['total_purchase'] < df2['total_purchase'].quantile(0.2)) |
        (df2['last_purchase_days'] > 180), 'customer_segment'] = 'At_Risk'
print(f"고객 세그먼트 분포 : \n{df2['customer_segment'].value_counts()}")
# 순서가 있는 범주형 데이터 (성별, 도시) -> 원-핫 인코딩
# city가 서울이면 city_seoul이 1, 나머지는 0
# df2_encoded = pd.get_dummies(df2, columns=['gender','city'], prefix=['gender', 'city'])
# print(f"인코딩 후 열 개수 : {len(df2_encoded.columns)}개")
# print(f"새로 생성된 열 : {[col for col in df2_encoded.columns if col not in df2.columns]}")
# print(df2_encoded.head())
# # 수치형 데이터 정규화
# numeric_columns = ['age', 'total_purchase', 'purchase_count', 'last_purchase_days', 'avg_purchase_amount']
# # StandardScaler 사용한 표준화
# scaler = StandardScaler()
# df2_scaled = df2_encoded.copy()
# df2_scaled[numeric_columns] = scaler.fit_transform(df2[numeric_columns])
# print("\n정규화 후 통계 : ", df2_scaled[numeric_columns].describe())
# print("\n=== 4단계 : 데이터 전처리 완료 리포트 ===")
# print(f"원본 데이터: {df2_original.shape[0]}행 {df2_original.shape[1]}열")
# print(f"최종 데이터: {df2_scaled.shape[0]}행 {df2_scaled.shape[1]}열")
# print(f"처리된 문제들 : ")
# print(f"    - 나이 이상치 {len(df2_original[(df2_original['age']<0) | (df2_original['age']> 100)])}개 수정")
# print(f"    - 성별 데이터 표준화 완료")
# print(f"    - 도시 결측치 {df2_original['city'].isnull().sum()}개 처리")
# print(f"    - 구매 금액 이상치 처리 완료")
# print(f"    - 파생 변수 2개 생성 (평균 구매 금액, 고객 세그먼트)")
# print(f"    - 범주형 데이터 인코딩 완료")
# print(f"    - 수치형 데이터 인코딩 완료")
# print(f"\n데이터 전처리가 성공적으로 완료되었습니다.")


# p.179 * 데이터 시각화 *
import matplotlib.pyplot as plt
# 선 그래프
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()
plt.rc('font', family='Apple SD Gothic Neo') # 한글폰트 적용
x = np.linspace(0, 10, 20) # 0부터 10까지 20개의 값 생성
plt.figure(figsize=(10, 6)) # 그래프 크기 설정
plt.plot(x, x, color='blue', linestyle='-', linewidth=2, marker='o',
         markersize=8, label='선형') # 그래프 설정 : x축 데이터, y축 데이터, 색깔, 선 스타일, 선 두께, 마커 스타일, 마커 크기, 범례
plt.plot(x, x**2, color='red', linestyle='--', linewidth=1.5, marker='s',
         markersize=6, label='제곱')
plt.grid(True) # 그리드 설정
plt.legend() # 범례 설정
plt.title('선 그래프 예제') # 제목 설정
plt.xlabel('x 값') # x축 레이블 설정
plt.ylabel('y 값') # y축 레이블 설정
plt.show() 
# 산점도
plt.scatter([1, 2, 3, 4], [1, 4, 9, 16], c='red', alpha=0.5)
plt.show()
np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=sizes, c=colors, alpha=0.7, cmap='viridis',
            marker='o', edgecolors='black') # 인자 : x축 데이터, y축 데이터, 크기, 색상, 투명도, 컬러 맵, 마커 스타일, 마커 테두리 색상
plt.colorbar(label='색상 값') # 색상 범례 설정
plt.grid(True)
plt.title('산점도 예제')
plt.xlabel('x 값')
plt.ylabel('y 값')
plt.show()
# 막대 그래프
plt.bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
plt.show()
categories = ['범주 A', '범주 B', '범주 C', '범주 D', '범주 E']
values = [5, 7, 3, 8, 6]
plt.figure(figsize=(10, 6))
plt.bar(categories, values, width=0.6, color=['#5DA5DA', '#FAA43A', '#60BD68',
                                               '#F17CB0', '#B2912F'],
        edgecolor='black', alpha=0.8, align='center') # 인자 : 범주, 값, 너비, 색상, 테두리 색상, 투명도, 정렬
plt.grid(True, axis='y')
plt.title('막대 그래프 예제')
plt.xlabel('범주')
plt.ylabel('값')
for i, v in enumerate(values):
    plt.text(i, v + 0.1, str(v), ha='center')
plt.show()
# 수평 막대 그래프
plt.figure(figsize=(10, 6))
plt.barh(categories, values, color='skyblue', edgecolor='black')
plt.title('수평 막대 그래프')
plt.xlabel('값')
plt.ylabel('범주')
plt.grid(True, axis='x')
plt.show()
# 히스토그램
plt.hist(np.random.normal(0, 1, 1000), bins=30)
plt.rc('axes', unicode_minus=False) # 유니코드 마이너스 대신 일반 하이픈 사용
plt.show()
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1.5, 1000)
plt.hist(data1, bins=30, alpha=0.7, color='blue', label='데이터셋 1',
         density=True, histtype='stepfilled', edgecolor='black') # 인자 : 데이터, 빈도, 투명도, 색상, 레이블 ,밀도, 히스토그램 유형, 테두리 색상
plt.hist(data2, bins=30, alpha=0.7, color='red', label='데이터셋 2',
         density=True, histtype='stepfilled', edgecolor='black')
plt.grid(True, alpha=0.3)
plt.title('히스토그램 예제 - 두 분포 비교')
plt.xlabel('값')
plt.ylabel('밀도')
plt.legend()
plt.axvline(np.mean(data1), color='blue', linestyle='dashed', linewidth=1) # 평균선 추가
plt.axvline(np.mean(data2), color='red', linestyle='dashed', linewidth=1)
plt.show()
# 누적 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(data1, bins=30, cumulative=True, histtype='step',
         linewidth=2, label='누적 분포')
plt.grid(True)
plt.title('누적 히스토그램')
plt.xlabel('값')
plt.ylabel('누적 빈도')
plt.legend()
plt.show()
# 파이 차트
plt.pie([15, 30, 45, 10], labels=['A', 'B', 'C', 'D'], autopct='%1.1f%%')
plt.show()
labels = ['제품 A', '제품 B', '제품 C', '제품 D', '기타']
sizes = [35, 25, 20, 15, 5]
explode = (0.1, 0, 0, 0, 0)
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
plt.figure(figsize=(10, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
plt.axis('equal')
plt.title('파이 차트 예제 - 제품별 시장 점유율')
plt.legend(loc='upper left')
plt.show()
# 도넛 차트
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
centre_circle = plt.Circle((0,0), 0.70, fc='white', edgecolor='black')
fig = plt.gcf() # 현재 그래프 객체 가져오기
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.title('도넛 차트 예제')
plt.show()

# * AI 기본 개념 *
# 머신러닝 예제) 집값 예측
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def house_price_prediction():
    np.random.seed(42)
    house_sizes = np.random.normal(100, 30, 1000)
    # print(house_sizes)
    house_prices = house_sizes * 50 + np.random.normal(0, 500, 1000) + 2000
    # print(house_prices)
    # 데이터 전처리
    X = house_sizes.reshape(-1, 1)
    y = house_prices
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 1, random_state=42
    )
    # 머신러닝 모델 생성 및 훈련
    model = LinearRegression()
    model.fit(X_train, y_train)
    # 예측
    y_pred = model.predict(X_test)
    # 성능 평가
    mse = mean_squared_error(y_test, y_pred) # 모델이 얼마나 예측을 잘하는지
    r2 = r2_score(y_test, y_pred) # 모델이 얼마나 변동성에 반응하는지
    print(f"평균 제곱 오차 (MSE) : {mse:.2f}")
    print(f"결정 계수(R²) : {r2:.2f}")
    print(f"모델 계수 (기울기) : {model.coef_[0]:.2f}")
    print(f"모델 절편 : {model.intercept_:.2f}")
    # 새로운 집 크기에 대한 예측
    new_house_sizes = [80, 120, 150]
    for size in new_house_sizes:
        predicted_price = model.predict([[size]])[0]
        print(f"{size}평 집의 예상 가격 : {predicted_price:.2f}만원")
    return model, X_test, y_test, y_pred

model, X_text, y_text, y_pred = house_price_prediction()

# 딥러닝 예제) mnist 손글씨 숫자를 인식
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def mnist_deep_learning():
    # 1. 데이터 로드 및 전처리
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    x_train_one = X_train[0]
    plt.imsave('x_train_one.png', x_train_one, cmap='gray')
    # 데이터 정규화
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0 # 데이터 개수는 자동으로 계산해서 28x28 픽셀 이미지를 1채널로 변환 -> 데이터 타입을 float32로 변환 -> 0~255 범위를 0~1 범위로 정규화
    print(X_train.shape) # 60000, 28, 28, 1 : 60000개의 28x28 픽셀 이미지 1채널, 각 픽셀은 4차원 배열
    print(X_train[0]) # 첫 번째 이미지 (28, 28, 1)
    print(X_train[0][10]) # 첫 번째 이미지의 10번째 행 (28, 1)
    print(X_train[0][10][5]) # 첫 번째 이미지의 10행 5열 픽셀 (1,)
    print(X_train[0][10][5][0]) # 해당 픽셀의 흑백 값
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')/255.0
    # 레이블 원-핫 인코딩
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 2. CNN 딥러닝 모델 구성 : 컨볼루션 레이어(Conv2D) -> 풀링 레이어(MaxPooling2D) -> 완전연결층(Dense)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # 첫 번째 컨볼루션 블록 -> 26x26x32
        MaxPooling2D((2, 2)), # 2x2 영역에서 최대값만 선택하여 크기를 절반으로 축소 -> 13x13x32
        Conv2D(64, (3, 3), activation='relu'), # 두 번째 컨볼루션 블록 -> 11x11x64
        MaxPooling2D((2, 2)), # 5x5x64
        Conv2D(64, (3, 3), activation='relu'), # 세 번째 컨볼루션 블록 -> 3x3x64
        Flatten(), # 3D 특징맵을 1D 벡터로 변환 -> 3x3x64 = 576차원 벡터로 평탄화
        Dense(64, activation='relu'), # 완전연결층 -> 576개를 64개로 압축
        Dropout(0.5), # 과적합 방지 기법
        Dense(10, activation='softmax') # 최종출력층 -> 10개의 클래스(0~9 숫자) 분류        
    ])

    # 3. 모델 컴파일(훈련 준비)
    model.compile(
        optimizer='adam', # Adam 최적화 알고리즘
        loss='categorical_crossentropy', # 다중 클래스 분류용 손실 함수(원-핫 인코딩된 라벨에 최적화)
        metrics=['accuracy'] # 훈련 중 모니터링할 지표(정확도)
    )

    # 4. 모델 훈련(학습 실행)
    print("모델 훈련 시작")
    print("훈련 과정 : CNN이 60,000개의 이미지에서 패턴을 학습")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=5,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 5. 모델 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n최종 테스트 정확도: {test_accuracy:.4f}")
    print(f"MNIST 벤치마크 : 일반적으로 99% 이상 달성 가능")

    # 6. 예측 예시
    sample_idx = np.random.randint(0, len(X_test), 5)
    predictions = model.predict(X_test[sample_idx])
    print("\n실제 예측 테스트 : ")
    for i, idx in enumerate(sample_idx):
        predicted_digit = np.argmax(predictions[i]) # 확률이 가장 높은 클래스 선택
        actual_digit = np.argmax(y_test[idx]) # 원-핫에서 실제 숫자 추출
        confidence = np.max(predictions[i]) * 100 # 최고 확률을 퍼센트로 변환(모델의 확신도)
        result = "✔️" if predicted_digit == actual_digit else "❌"
        print(f"샘플 {i+1} : 예측={predicted_digit}, 실제={actual_digit}, 신뢰도={confidence:.1f}%")
    print("\nCNN 모델 완성! 손글씨 숫자를 높은 정확도로 인식할 수 있습니다.")
    
    return model, history

model, history = mnist_deep_learning()

# * 삼각함수 기초 *
# 삼각함수 예제) 시계 만들기
import time

def create_clock():
    current_time = time.localtime()
    hour = current_time.tm_hour % 12
    minute = current_time.tm_min 
    second = current_time.tm_sec
    # 각도
    hour_angle = -2 * np.pi * (hour + minute/60)/12 + np.pi/2
    minute_angle = -2 * np.pi * minute/60 + np.pi/2
    second_angle = -2 * np.pi * second/60 + np.pi/2
    # 바늘의 끝점 좌표
    hour_x = 0.5 * np.cos(hour_angle)
    hour_y = 0.5 * np.sin(hour_angle)
    minute_x = 0.8 * np.cos(minute_angle)
    minute_y = 0.8 * np.sin(minute_angle)
    second_x = 0.9 * np.cos(second_angle)
    second_y = 0.9 * np.sin(second_angle)
    # 시계 시각화
    plt.rcParams['font.family'] = ['Apple SD Gothic Neo', 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(8,8))
    circle = plt.Circle((0,0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    for i, (hour_num, angle) in enumerate([(12, np.pi/2),(3, 0), (6, -np.pi/2), (9, np.pi)]):
        x = 1.1 * np.cos(angle)
        y = 1.1 * np.sin(angle)
        ax.text(x, y, str(hour_num), ha='center', va='center', fontsize=14, fontweight='bold')

    ax.plot([0, hour_x], [0, hour_y], 'b-', linewidth=6, label=f'시침 ({hour}시)')
    ax.plot([0, minute_x], [0, minute_y], 'g-', linewidth=3, label=f'시침 ({hour}분)')
    ax.plot([0, second_x], [0, second_y], 'r-', linewidth=1, label=f'시침 ({hour}초)')

    ax.plot(0, 0, 'ko', markersize=8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"삼각함수 시계\n현재 시각 : {hour:02d}:{minute:02d}:{second:02d}", fontsize=16)

    plt.show()

create_clock()

# * 지수 함수 *
from matplotlib import font_manager

plt.rcParams['axes.unicode_minus'] = False

def visualize_exponential_function():
    x = np.linspace(-3, 3, 1000)
    y1 = 2**x
    y2 = 3**x
    y3 = np.exp(x)
    y4 = (0.5)**x
    y5 = (0.3)**x

    plt.figure(figsize=(12,8))
    # 증가하는 지수함수
    plt.plot(x, y1, 'r-', linewidth=2, label='2^x')
    plt.plot(x, y2, 'g-', linewidth=2, label='3^x')
    plt.plot(x, y3, 'b-', linewidth=2, label='e^x')
    # 감소하는 지수함수
    plt.plot(x, y4, 'r--', linewidth=2, label='(1/2)^x')
    plt.plot(x, y5, 'g--', linewidth=2, label='(1/3)^x')
    # 특별한 점 표시
    plt.plot(0, 1, 'ko', markersize=8, label='Common Point (0,1)')
    # x축, y축 표시
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.7)
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)
    # 그래프 설정
    plt.xlim(-3, 3)
    plt.ylim(-0.5, 8)
    plt.xlabel('x', fontsize=12)
    plt.xlabel('f(x)', fontsize=12)
    plt.title('Exponential Functions Comparison\n'
              'Solid lines: a>1 (increasing)\n'
              'Dashed lines: 0<a<1 (decreasing)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()