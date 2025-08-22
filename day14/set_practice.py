# p.162
# 1. 목표 : 소셜 네트워크에서 사용자 간의 관계와 추천 시스템을 구현하는 프로그램을 작성
# 2. 요구사항
# 공통 관심사를 갖는 친구 응답
# 공통 관심사가 없는 친구 응답

users = {
    "Alice": ["음악", "영화", "독서"],
    "Bob": ["스포츠", "여행", "음악"],
    "Charlie": ["프로그래밍", "게임", "영화"],
    "David": ["요리", "여행", "사진"],
    "Eve": ["프로그래밍", "독서", "음악"],
    "Frank": ["스포츠", "게임", "요리"],
    "Grace": ["영화", "여행", "독서"]
}

# 입력받은 유저와 비교해서 공통 관심사가 있으면 사용자 이름과 공통 관심사 출력
def common_interest(user_name, show=True):
    common_users = []
    user_set = set(users[user_name])

    for user in users:
        if user != user_name:
            check_user_set = set(users[user])
            common = user_set & check_user_set
            if common:
                common_users.append((user, common))
    
    if not common_users:
        print("공통 관심사를 가진 사용자가 없어요.ㅜㅜ")
    else:
        for user, common in common_users:
            print(f"{user}님과 공통 관심사가 있어요.")
            print(f"* 공통 관심사 : {", ".join(common)}") 
            print()

    return common_users    

# 입력받은 유저와 비교해서 공통 관심사가 없으면 사용자 이름 출력
def not_common_interest(user_name):
    not_common_users = []
    user_set = set(users[user_name])

    for user in users:
        if user != user_name:
            check_user_set = set(users[user])
            not_common = user_set.isdisjoint(check_user_set)
            if not_common:
                not_common_users.append((user))
    
    for user in not_common_users:
        print(f"{user}님과 공통 관심사가 없어요.")
        print(f"{user}님의 관심사는 {", ".join(check_user_set)}입니다.")
        print()

# 입력받은 유저와 특정 사용자를 입력받아서 서로 공통 관심사가 뭐가 있는지 출력
def compare_common_interest(user_name):
    print("*** 등록된 사용자 목록 ***")
    for user in users:
        if user != user_name:
            print(user)
    
    print()
    while True:   
        search_user_name = input("=> 비교할 사용자 입력 : ")
        if search_user_name in users:
            break
        else:
            print("등록된 사용자가 없습니다.")
    print()

    common = set(users[user_name]) & set(users[search_user_name])
    if common:
        print(f"두 분의 공통 관심사는 {", ".join(common)}입니다.")
    else:
        print("두 분의 공통 관심사가 없어요!")

# 사용자 추천
def recommend_user(user_name):
    commons = common_interest(user_name, show=False)

    if not commons:
        print("공통 관심사를 가진 사용자가 없어요.ㅜㅜ")
    
    ranked = sorted([(user, common, len(common)) for user, common in commons], reverse=True)
    top_user = [name for name, count in ranked if count == max(count for name, count in ranked)]
    
    print(f"당신과 가장 관심사가 비슷한 사용자는 : {", ".join(top_user)}입니다.")

while True:   
    user_name = input("사용자 이름 : ")
    if user_name in users:
        break
    else:
        print("등록된 사용자가 없습니다.")

while True:
    print()
    print("====== 원하는 작업을 선택하세요 ======")
    print("1. 나랑 관심사가 통하는 사람")
    print("2. 나랑 관심사가 통하지 않는 사람")
    print("3. 이 사람과 내가 서로 공통 관심사가 있을까?")
    print("4. 사용자 추천 받기")
    print("5. 종료")
    print()

    choice = input("번호 : ")

    print()
    print(choice+"번을 선택하였습니다.")
    print()

    if choice == "1":
        common_interest(user_name)
    elif choice == "2":
        not_common_interest(user_name)
    elif choice == "3":
        compare_common_interest(user_name)
    elif choice == "4":
        recommend_user(user_name)
    elif choice == "5":
        print("종료합니다.")
        break
    else:
        print("잘못누르셨습니다.")
