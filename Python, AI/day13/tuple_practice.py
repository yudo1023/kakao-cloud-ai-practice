# p.123
# 1. 목표 : 주어진 데이터셋에서 튜플을 활용하여 분석
# 2. 요구사항
# 연도별 판매량 계산
# 제품별 평균 가격 계산
# 최대 판매 지역 찾기
# 분기별 매출 분석
# ==데이터셋==
# 데이터: 연도, 분기, 제품, 가격, 판매량, 지역
sales_data = [
    (2020, 1, "노트북", 1200, 100, "서울"),
    (2020, 1, "스마트폰", 800, 200, "부산"),
    (2020, 2, "노트북", 1200, 150, "서울"),
    (2020, 2, "스마트폰", 800, 250, "대구"),
    (2020, 3, "노트북", 1300, 120, "인천"),
    (2020, 3, "스마트폰", 850, 300, "서울"),
    (2020, 4, "노트북", 1300, 130, "부산"),
    (2020, 4, "스마트폰", 850, 350, "서울"),
    (2021, 1, "노트북", 1400, 110, "대구"),
    (2021, 1, "스마트폰", 900, 220, "서울"),
    (2021, 2, "노트북", 1400, 160, "인천"),
    (2021, 2, "스마트폰", 900, 270, "부산"),
    (2021, 3, "노트북", 1500, 130, "서울"),
    (2021, 3, "스마트폰", 950, 320, "대구"),
    (2021, 4, "노트북", 1500, 140, "부산"),
    (2021, 4, "스마트폰", 950, 370, "서울")
]

# 연도별 판매량 계산
def SalesByYear():
    year_sorted = sorted(sales_data, key=lambda x:x[0])

    sales_by_year = []
    # pre_year = None
    pre_year = year_sorted[0][0]
    year_sum = 0

    for row in year_sorted:
        cur_year = row[0]
        qty = row[4]

        # if pre_year == None:
        #     pre_year = cur_year

        if cur_year == pre_year:
            year_sum += qty
        else:
            sales_by_year.append((pre_year, year_sum))
            year_sum = qty
            pre_year = cur_year

    sales_by_year.append((pre_year, year_sum))

    sales_by_year = tuple(sales_by_year)
    # print(f"연도별 판매량(연도, 판매량) : {sales_by_year}")
    print("***연도별 판매량***")
    # for item in sales_by_year:
    #     print(f"{item[0]}년 판매량 : {item[1]}")
    for year, total in sales_by_year:
        print(f"{year}년 판매량 : {total}")
    print() # print("\n")
    

# 제품별 평균 가격 계산
def ProductAveragePrice():
    product_sorted = sorted(sales_data, key=lambda x:x[2])

    product_average_price = []
    # pre_product = None
    pre_product = product_sorted[0][2]
    price_sum = 0
    count = 0
    price_avg = 0

    for row in product_sorted:
        cur_product = row[2]
        price = row[3]

        # if pre_product == None:
        #     pre_product = cur_product

        if cur_product == pre_product:
            price_sum += price
            count += 1
        else:
            price_avg = price_sum / count
            product_average_price.append((pre_product, price_avg))
            pre_product = cur_product
            price_sum = price
            count = 1

    price_avg = price_sum/count
    product_average_price.append((pre_product, price_avg))

    product_average_price = tuple(product_average_price)
    # print(f"제품별 평균 가격(제품, 평균 가격) : {product_average_price}")
    print(f"***제품별 평균 가격***")
    for product, avg_price in product_average_price:
        print(f"제품명 : {product} / 평균 가격 : {avg_price:.2f}")
    print()

# 최대 판매 지역 찾기
def MaxSalesByRegion():
    region_sorted = sorted(sales_data, key=lambda x:x[5])

    sales_by_region = []
    # pre_region = None
    pre_region = region_sorted[0][5]
    region_sum = 0

    for row in region_sorted:
        cur_region = row[5]
        qty = row[4]

        # if pre_region == None:
        #     pre_region = cur_region

        if cur_region == pre_region:
            region_sum += qty
        else:
            sales_by_region.append((pre_region, region_sum))
            region_sum = qty
            pre_region = cur_region

    sales_by_region.append((pre_region, region_sum))

    max_sales_by_region = max((sales_by_region), key=lambda x:x[1])
    # print(f"최대 판매 지역(지역, 판매량) : {max_sales_by_region}")
    print(f"***최대 판매 지역***")
    print(f"지역 : {max_sales_by_region[0]} / 판매량 : {max_sales_by_region[1]}")
    print()

# 분기별 매출 분석
def SalesByBranch():
    branch_sorted = sorted(sales_data, key=lambda x:x[1])

    sales_by_branch = []
    # pre_branch = None
    pre_branch = branch_sorted[0][1]
    branch_sum = 0

    for row in branch_sorted:
        cur_branch = row[1]
        qty = row[3] * row[4]

        # if pre_branch == None:
        #     pre_branch = cur_branch

        if cur_branch == pre_branch:
            branch_sum += qty
        else:
            sales_by_branch.append((pre_branch, branch_sum))
            branch_sum = qty
            pre_branch = cur_branch

    sales_by_branch.append((pre_branch, branch_sum))

    sales_by_branch = tuple(sales_by_branch)
    # print(f"분기별 매출(분기, 매출) : {sales_by_branch}")
    print(f"***분기별 매출***")
    for branch, total in sales_by_branch:
        print(f"{branch}분기 판매량 : {total}")
    print()
    
SalesByYear()
ProductAveragePrice()
MaxSalesByRegion()
SalesByBranch()