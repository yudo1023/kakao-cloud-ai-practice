import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# * NumPy *
# 1차원 배열
arr1 = np.array([1,2,3,4,5])
print(arr1)
# 2차원 배열
arr2 = np.array([[1, 2, 3], [4, 5, 6]]) 
print(arr2)
# 0으로 채워진 배열
zeros = np.zeros((3, 4)) # 3행 4열
print(zeros)
# 1로 채워진 배열
ones = np.ones((2, 3)) # 2행 3열
print(ones)
# 특정 범위의 균일한 간격 배열
range_arr = np.arange(0, 10, 2) # 0에서 10까지 2 간격으로
print(range_arr)
# 선형 간격 배열
linear_space = np.linspace(0, 1, 5) # 0과 1 사이를 5등분
print(linear_space)
# 랜던 배열
random_arr = np.random.random((2, 2)) # 2x2 균등분포 난수 행렬
print(random_arr)
# 배열 속성
print("배열 차원 : ",arr2.ndim)
print("배열 형태 : ",arr2.shape)
print("배열 크기 : ",arr2.size)
print("요소 데이터 타입 : ",arr2.dtype)
print("각 요소 바이트 크기 : ",arr2.itemsize)
print("전체 배열 바이트 크기 : ",arr2.nbytes)
# 전치(행과 열 바꾸기)
print("원본 배열 : ",arr2)
print("전치 배열 (T) : ",arr2.T)
# 배열 형태 변경
arr1d1 = np.arange(12)
arr2d1 = arr1d1.reshape(3, 4)
print("reshape 결과 : ",arr2d1)
# 배열 평탄화(1차원으로 변환)
print("평탄화 결과 (flattern) : ",arr2d1.flatten())
# 데이터 타입 변환
arr_float = arr2.astype(np.float64)
print("타입 변환 후 : ",arr_float.dtype)
# 통계 메서드
data = np.array([1, 2, 3, 4, 5])
print("합계 : ", data.sum())
print("평균 : ",data.mean())
print("최소값 : ",data.min())
print("최대값 : ",data.max())
print("표준편차 : ",data.std())
print("분산 : ",data.var())
print("누적합 : ",data.cumsum())
# 조건 기반 인덱싱
arr3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("짝수 요소만 선택 : ",arr3[arr3 % 2 == 0])
# 배열 연산 메서드
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("행렬 곱셈 (dot) : ", a.dot(b)) # 수학적 행렬 곱셈 규칙에 따라 계산
print("요소별 곱셈 (*) : ", a * b) # 대응하는 위치의 요소끼리의 곱
# 배열 인덱싱
# 1차원 배열 인덱싱
arr1d2 = np.array([10, 20, 30, 40, 50])
print(arr1d2[0]) # 첫 번째 요소
print(arr1d2[-1]) # 마지막 요소
# 2차원 배열 인덱싱
arr2d2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d2[0, 0]) # 첫 번째 행, 첫 번쨰 열
print(arr2d2[1, 2]) # 두 번째 행, 세 번쨰 열
print(arr2d2[2, -1]) # 세 번쨰 행, 마지막 열
# 배열 슬라이싱
# 1차원 배열 슬라이싱
arr1d3 = np.array([10, 20, 30, 40, 50, 60])
print(arr1d3[1:4]) # 인데스 1부터 3까지
print(arr1d3[:3]) # 처음부터 인데스 2까지
print(arr1d3[3:]) # 인덱스 3부터 끝까지
print(arr1d3[::2]) # 처음부터 끝까지 2 간격으로
# 2차원 배열 슬라이싱
arr2d3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(arr2d3[:2, 1:3]) # 처음 2행, 1~2열
print(arr2d3[1:, :2]) # 1행부터 끝까지, 0~1열
# 브로드캐스팅
a = np.array([[1, 2, 3],[4, 5, 6]])
b = np.array([10, 20, 30])
print(a+b)
celsius = np.array([0, 15, 30 ,45])
fahrenheit = celsius * 9/5 + 32
print(fahrenheit)
# 배열 변형 및 조작
reshaped1 = arr1d1.reshape(2, 6)
print(reshaped1)
reshaped2 = arr1d1.reshape(3, 4)
print(reshaped2)
reshaped3 = arr1d1.reshape(3, -1)
print(reshaped3)
# 배열 합치기
a2 = np.array([[1, 2],[3, 4]])
b2 = np.array([[5, 6],[7, 8]])
# 세로 방향으로 합치기 (행 추가)
vertical = np.concatenate([a2, b2], axis=0)
print(vertical)
# 가로 방향으로 합치기 (열 추가)
horizontal = np.concatenate([a2, b2], axis=1)
print(horizontal)
# vstack, hstack 함수
v_stack = np.vstack([a2, b2])
h_stack = np.hstack([a2, b2])
print(v_stack)
print(h_stack)
# 배열 분할
arr4 = np.arange(12).reshape(3, 4)
print(arr4)
# 수평 분할(행 기준)
h_split = np.split(arr4, 3, axis=0)
for i, split_arr in enumerate(h_split):
    print("분할 {i} :", split_arr)
# 수직 분할(열 기준)
v_split = np.split(arr4, 2, axis=1)
for i, split_arr in enumerate(v_split):
    print("분할 {i} :", split_arr)
# 데이터 필터링
ages = np.array([23, 18, 45, 61, 17, 34, 57, 28, 15, 42])
# 조건식 필터링
adult_filter = ages >= 18
print(adult_filter)
# 불리언 인덱싱
adults = ages[adult_filter]
print(adults)
young_adults = ages[(ages >= 18) & (ages < 30)]
print(young_adults)
ticket_prices = np.zeros_like(ages) # 같은 크기의 0으로 채워진 배열 생성
# 조건부 값 할당
ticket_prices[ages < 18] = 5
ticket_prices[(ages >= 18) & (ages < 60)] = 10
ticket_prices[ages >= 60] = 8
print(ticket_prices)
# 이미지 필터링
image = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,1,0,1,0],
    [0,1,1,1,0],
    [0,0,0,0,0],
])

plt.figure(figsize=(8,4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('original')

brightend = image + 0.5
plt.subplot(1,3,2)
plt.imshow(brightend, cmap='gray', vmin=0, vmax=1)
plt.title('brightend')

inverted = 1 - image
plt.subplot(1,3,3)
plt.imshow(inverted, cmap='gray', vmin=0, vmax=1)
plt.title('inverted')

plt.tight_layout()
plt.savefig('image_example.png')
plt.show()

import pandas as pd
# * Pandas *
# 기본 series 생성
s1 = pd.Series([1, 3, 5, 7, 9])
print(s1)
# 인덱스 지정 Series 생성
s2 = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(s2)
# 딕셔너리로 Series 생성
population = {
    'Seoul' : 9776,
    'Busan' : 3429,
    'Incheon' : 2947,
    'Daegu' : 2465
}
pop_series = pd.Series(population)
print(pop_series)
# 기본 속성
print("값 배열 : ", s2.values)
print("인덱스 : ", s2.index)
# 기본 통계 메소드
print("평균 : ", s2.mean())
print("합계 : ", s2.sum())
print("최소값 : ", s2.min())
print("최대값 : ", s2.max())
# 데이터 접근
print("'c' 인덱스의 값 : ", s2['c'])
print("여러 인덱스의 값 : ", s2[['a', 'c', 'e']])
# 조건부 필터링
print("30보다 큰 값 : ", s2[s2 > 30])
# 데이터 변환
print("제곱근 : ", s2.apply(np.sqrt)) # 각 요소에 대한 연산 수행
# 결측지 확인 및 처리
print("결측치 여부 : ", s2.isna()) # 결측치 여부를 불리언으로 반환
print("결측치 제외 : ", s2.dropna()) # 결측치가 있는 항목 제거
print("결측치 0으로 채우기 : ", s2.fillna(0)) # 결측치를 0으로 대체
# DataFrame
data = {
    'Name' : ['John', 'Anna', 'Peter', 'Linda', 'Bob'],
    'Age' : [28, 24, 35, 32, 45],
    'City' : ['New York', 'Paris', 'Berlin', 'London', 'Tokyo'],
    'Salary' : [50000, 65000, 75000, 85000, 60000],
    'Department' : ['IT', 'HR', 'IT', 'Finance', 'Marketing']
}
df = pd.DataFrame(data)
# 기본 속성
print("크기(행, 열) : ", df.shape)
print("열 이름 : ", df.columns)
print("행 인덱스 : ", df.index)
print("데이터 타입 : ", df.dtypes)
# 데이터 확인
print("처음 2행\n : ", df.head(2))
print("마지막 2행\n : ", df.tail(2))
print("기본 통계량\n : ", df.describe())
# 데이터 접근
print("'Age' 열 : \n", df['Age'])
print("여러 열 선택 : \n", df[['Name', 'Salary']])
print("첫 3행 : \n", df.iloc[0:3])
print("인덱스 1, 3, 4행 선택 : \n", df.loc[[1,3,4]])
print("첫 2행의 'Name'과 'Age' 열 : \n", df.loc[0:1, ['Name', 'Age']])
print("조건부 선택 : \n", df[df['Age'] > 30])
print("값 존재 여부 필터링 : \n", df[df['City'].isin(['Tokyo', 'London'])])
# 데이터 수정
df['Age'] = df['Age'] + 1
print("모든 나이에 1 추가 : \n", df)
df['Country'] = ['USA', 'France', 'Germany', 'UK', 'Japan']
print("새 열 추가 : \n", df)
df.loc[6] = ['Charlie', 29, 'Sydney', 70000, 'Finance', 'Australia']
print("새 행 추가 : \n", df)
df.drop('Country', axis=1, inplace=True)
df.drop(6, axis=0, inplace=True)
print("열 삭제: \n", df)
print("행 삭제: \n", df)
# DataFrame GroupBy
df2 = pd.DataFrame({
    'Department' : ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT'],
    'Employee' : ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Salary' : [75000, 65000, 80000, 90000, 60000, 78000],
    'Age' : [28, 35,32, 45, 30, 29],
    'Year' : [2021, 2022, 2021, 2022, 2021, 2022]
})
# 기본 연산
dept_groups = df2.groupby('Department')
print("부서별 평균 급여 : ", dept_groups['Salary'].mean())
# 다중 열 그룹화
year_dept_groups = df2.groupby(['Year', 'Department'])
print("연도별, 부서별 평균 급여 : ", year_dept_groups['Salary'].mean())
# 기본 통계
print("부서별 급여 통계 요약 : ", dept_groups['Salary'].describe())
# 집계 함수
print("여러 집계 함수 적용 : ", dept_groups['Salary'].agg(['count', 'mean', 'sum', 'std', 'min', 'max']))
print("열별 다른 집계 함수 : \n", dept_groups.agg({
    'Salary' : ['mean', 'max'],
    'Age' : ['mean', 'min', 'max']
}))
# transform : 원본 데이터프레임의 인덱스와 크기를 그대로 유지하면서 그룹 연산 결과 반환
df2['Dept_Avg_Salary'] = dept_groups['Salary'].transform('mean')
print("각 직원의 급여와 부서 평균 급여 : ", df2[['Employee', 'Department', 'Salary', 'Dept_Avg_Salary']])
# filter : 그룹 조건에 따라 필터링
high_salary_depts = dept_groups.filter(lambda x: x['Salary'].mean() > 70000)
print("평균 급여가 70000 이상인 부서의 모든 직원 : ", high_salary_depts)
# get_gorup : 특정 그룹 선택
print("IT 부서 직원 : ", dept_groups.get_group('IT'))
# 시간 기반 그룹화 -> dt 접근자 사용
df2['Date'] = pd.date_range(start='2022-01-01', periods=len(df2), freq='M')
print("월별 평균 급여 : ", df2.groupby(df2['Date'].dt.month)['Salary'].mean())
print("분기별 평균 급여 : ", df2.groupby(df2['Date'].dt.quarter)['Salary'].mean())
# 연속 변수의 범주화 후 그룹화 -> pd.cut()
df2['Age_Group'] = pd.cut(df2['Age'], bins=[20, 30, 40, 50], labels=['20대', '30대', '40대'])
print("연령대별 평균 급여 : ", df2.groupby('Age_Group')['Salary'].mean())
# 크기 기반 분위수로 그룹화 -> pd.qcut()
df2['Salary_Quantile'] = pd.qcut(df2['Salary'], q=3, labels=['Low', 'Medium', 'High'])
print("급여 분위수별 평균 나이 : ", df2.groupby('Salary_Quantile')['Age'].mean())
# 사용자 정의 함수로 그룹화 -> apply()
def experience_level(age):
    """나이를 기준으로 경력 수준을 분류하는 함수"""
    if age < 30:
        return 'Junior'
    elif age < 40:
        return 'Mid-level'
    else:
        return 'Senior'
df2['Experience'] = df2['Age'].apply(experience_level)
print("경력 수준별 평균 급여 : ", df2.groupby('Experience')['Salary'].mean())
# 정렬 메서드
df3 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Score': [85, 92, 78, 96, 88, 73],
    'Attendance': [95, 80, 90, 75, 85, 92]
})
print("점수 기준 상위 3명 : ", df3.nlargest(3, 'Score'))
print("출석률 기준 하위 2명 : ", df3.nsmallest(2, 'Attendance'))
print("점수와 출석률 모두 높은 상위 3명 : ", df3.nlargest(3, ['Score', 'Attendance']))
print("점수 기준 내림차순 정렬 : ", df3.sort_values('Score', ascending=False))
print("여러 열 기준 정렬(점수 내림차순, 출석률 오름차순 정렬) : ", df3.sort_values(['Score', 'Attendance'], ascending=[False, True]))
# 상관관계 및 공분산
print("점수와 출석률의 상관관계 : ", df3[['Score', 'Attendance']].corr())
print("점수와 출석률의 공분산 : ", df3[['Score', 'Attendance']].cov())
# 데이터 병합
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com',
              'david@example.com', 'eve@example.com']
})
orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 2, 3, 4, 5],
    'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
    'amount': [1200, 800, 450, 300, 80]
})
print("고객 정보 : ", customers)
print("주문 정보 : ", orders)
inner_join = pd.merge(customers, orders, on='customer_id')
print("내부 조인(고객, 주문 모두 있는 경우) : ", inner_join)
left_join = pd.merge(customers, orders, on='customer_id', how='left')
print("왼쪽 조인(모든 고객, 주문 없으면 NaN) : ", left_join)
right_join = pd.merge(customers, orders, on='customer_id', how='right')
print("오른쪽 조인(모든 주문, 고객 없으면 NaN) : ", right_join)
outer_join = pd.merge(customers, orders, on='customer_id', how='outer')
print("외부 조인(모든 고객 및 주문) : ", outer_join)
# 데이터 연결
df4 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
})
df5 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5']
})
result_rows = pd.concat([df4, df5])
print("세로 연결(행 추가) : ", result_rows)
df6 = pd.DataFrame({
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
})
result_cols = pd.concat([df4, df6])
print("가로 연결(열 추가) : ", result_cols)

# * 데이터 전처리 *
# 1. 결측치 처리
df7 = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, np.nan]
})
print("원본 데이터 : ", df7)
print("결측지 여부 : ", df7.isna())
print("열별 결측지 여부 : ", df7.isna().sum())
print("결측지 행 삭제 후 : ", df7.dropna())
print("결측지 0으로 채운 후: ", df7.fillna())
print("결측지 평균으로 채운 후 : ", df7.filna(df7.mean()))
# 2. 데이터 타입 변환
df8 = pd.DataFrame({
    'A': ['1', '2', '3', '4', '5'],
    'B': [1.1, 2.2, 3.3, 4.4, 5.5],
    'C': ['2020-01-01', '2020-02-01','2020-03-01'
          '2020-04-01','2020-05-01'],
    'D': ['True', 'False', 'True', 'False', 'True']
})
print("원본 데이터 : ", df8)
# 문자열 -> 정수 변환
df8['A'] = df8['A'].astype(int)
# 문자열 -> 날짜/시간 변환
df8['C'] = pd.to_datetime(df8['C'])
# 문자열 -> 불리언 변환
df8['D'] = df8['D'].astype(bool) # 문자열 'False'도 True로 변환됨(주의), 잘못된 방법
df8['D_correct1'] = df8['D'].map({'True': True, 'False': False}) # 올바른 방법1, map()사용
df8['D_correct2'] = (df8['D'] == 'True') # 올바른 방법2, 조검문 사용
# 3. 이상치 탐지 및 제거
np.random.seed(42)
normal_data = np.random.normal(50, 10, 95)   # 정상 데이터 95개
outliers = [120, 130, -20, -10, 150]         # 이상치 5개
data_with_outliers = np.concatenate([normal_data, outliers])
df = pd.DataFrame({
    'ID': range(1, 101),
    'Score': data_with_outliers,
    'Category': np.random.choice(['A', 'B', 'C'], 100)
})
print("원본 데이터 통계:")
print(df['Score'].describe())
# IQR 방법
def detect_outliers_iqr(data):
    """IQR 방법으로 이상치 탐지"""
    Q1 = data.quantile(0.25)   # 1사분위수
    Q3 = data.quantile(0.75)   # 3사분위수
    IQR = Q3 - Q1              # 사분위수 범위
    # 이상치 경계값 계산 (일반적으로 1.5 * IQR 사용)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 이상치 식별
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, lower_bound, upper_bound
outliers_mask, lower, upper = detect_outliers_iqr(df['Score'])
print(f"\nIQR 방법 - 이상치 경계: {lower:.2f} ~ {upper:.2f}")
print(f"이상치 개수: {outliers_mask.sum()}개")
print("이상치 값들:")
print(df[outliers_mask]['Score'].values)
# Z-Score 방법
def detect_outliers_zscore(data, threshold=3):
    """Z-Score 방법으로 이상치 탐지 (기본 임계값: 3)"""
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers = z_scores > threshold
    return outliers, z_scores
outliers_zscore, z_scores = detect_outliers_zscore(df['Score'])
print(f"\nZ-Score 방법 - 이상치 개수: {outliers_zscore.sum()}개")
print("Z-Score가 3 이상인 값들:")
print(df[outliers_zscore][['ID', 'Score']].values)
# 이상치 제거
df_no_outliers = df[~outliers_mask].copy()  # IQR 방법으로 탐지된 이상치 제거
print(f"\n이상치 제거 전 데이터 크기: {len(df)}")
print(f"이상치 제거 후 데이터 크기: {len(df_no_outliers)}")
print("\n이상치 제거 후 통계:")
print(df_no_outliers['Score'].describe())
# 이상치를 중앙값으로 대체
df_replaced = df.copy()
median_score = df['Score'].median()
df_replaced.loc[outliers_mask, 'Score'] = median_score
print(f"\n이상치를 중앙값({median_score:.2f})으로 대체 후 통계:")
print(df_replaced['Score'].describe())
# 4. 중복 데이터 제거
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Bob', 'Eve', 'Charlie'],
    'Age': [25, 30, 35, 25, 40, 30, 28, 35],
    'City': ['Seoul', 'Busan', 'Seoul', 'Seoul', 'Daegu', 'Busan', 'Seoul', 'Seoul'],
    'Salary': [50000, 60000, 70000, 50000, 80000, 65000, 55000, 70000]
}
df = pd.DataFrame(data)
print("원본 데이터:")
print(df)
print(f"\n원본 데이터 크기: {len(df)}행")
# 완전 중복 행 확인
print("\n=== 완전 중복 행 탐지 ===")
duplicate_rows = df.duplicated()  # 모든 열이 동일한 행 탐지
print("중복 여부:\n", duplicate_rows)
print(f"완전 중복 행 개수: {duplicate_rows.sum()}개")
if duplicate_rows.sum() > 0:
    print("중복된 행들:")
    print(df[duplicate_rows])
# 특정 열 기준 중복 확인
print("\n=== 특정 열 기준 중복 탐지 ===")
name_duplicates = df.duplicated(subset=['Name'])
print("이름 기준 중복 행:")
print(df[name_duplicates])
name_age_duplicates = df.duplicated(subset=['Name', 'Age'])
print("\n이름+나이 기준 중복 행:")
print(df[name_age_duplicates])
# 중복 데이터 제거
# 완전 중복 행 제거
df_no_duplicates = df.drop_duplicates()
print(f"\n완전 중복 제거 후: {len(df_no_duplicates)}행")
print(df_no_duplicates)
# 특정 열 기준 중복 제거
df_unique_names = df.drop_duplicates(subset=['Name'])
print(f"\n이름 기준 중복 제거 후: {len(df_unique_names)}행")
print(df_unique_names)
# 조건부 중복 제거
print("\n=== 조건부 중복 제거 (최고 급여 유지) ===")
df_max_salary = df.loc[df.groupby('Name')['Salary'].idxmax()]
print("각 이름별 최고 급여 데이터만 유지:")
print(df_max_salary.sort_values('Name'))
# 중복 데이터 통계 요약
print("\n=== 중복 데이터 요약 통계 ===")
total_rows = len(df)
unique_rows = len(df.drop_duplicates())
duplicate_count = total_rows - unique_rows
duplicate_percentage = (duplicate_count / total_rows) * 100
print(f"전체 행 수: {total_rows}")
print(f"고유 행 수: {unique_rows}")
print(f"중복 행 수: {duplicate_count} ({duplicate_percentage:.2f}%)")
