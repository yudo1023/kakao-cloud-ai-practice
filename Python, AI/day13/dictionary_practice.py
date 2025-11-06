# p.145
# 1. 목표 : 딕셔너리를 활용한 간단한 주소록 프로그램 작성
# 2. 요구사항
# 연락처 이름을 키로 하고 전화번호, 이메일, 주소 등의 정보를 값으로 저장
# 중첩 딕셔너리 구조를 사용하여 각 연락처마다 여러 정보를 저장
# 연락처 추가, 삭제, 검색, 수정, 모든 연락처 보기 기능 구현

contacts = {
    "홍길동": {
        "phone": "010-1111-1111",
        "email": "hong@example.com",
        "address": "서울시"
    },
    "김철수": {
        "phone": "010-2222-2222",
        "email": "kim@example.com",
        "address": "부산시"
    }
}

# 연락처 추가
def add_contact(name, phone, email, address):
    contacts[name] = {"phone": phone, "email": email, "address": address}
    print("추가했습니다.")

# 연락처 삭제
def del_contact(name):
    if name in contacts:
        del contacts[name]
        print("삭제했습니다.")
    else:
        print("연락처에 없습니다.")

# 연락처 수정
def mod_contact(name):
    if name not in contacts:
        print("연락처에 없습니다.")
        return
    
    print("수정할 항목을 선택하세요 : (1) 전화번호 (2) 이메일 (3) 주소 (4) 취소")
    mod_choice = input("번호 : ")

    if mod_choice == "1":
        print(f"현재 전화번호 : {contacts[name]["phone"]}")
        new_phone = input("새 전화번호 : ")
        contacts[name]["phone"] = new_phone
    elif mod_choice == "2":
        print(f"현재 이메일 : {contacts[name]["email"]}")
        new_email = input("새 이메일 : ")
        contacts[name]["email"] = new_email
    elif mod_choice == "3":
        print(f"현재 주소 : {contacts[name]["address"]}")
        new_address = input("새 주소 : ")
        contacts[name]["address"] = new_address
    elif mod_choice == "4":
        print("취소하였습니다.")
        return
    else:
        print("잘못 선택했습니다.")
        return

    print("수정했습니다.")

# 연락처 검색
def search_contact(name):
    if name in contacts:
        info = contacts[name]
        print(f"[{name}] (1) 전화번호 : {info['phone']} (2) 이메일 : {info['email']} (3) 주소 : {info['address']}")
    else:
        print("연락처에 없습니다.")

# 연락처 목록 보기
def view_contacts():
    if not contacts:
        print("연락처가 없습니다.")
        return
    print("*** 연락처 목록 ***")
    for name, info in contacts.items():
        print(f"[{name}] (1) 전화번호 : {info['phone']} (2) 이메일 : {info['email']} (3) 주소 : {info['address']}")
    print()

while True:
    print()
    print("====== 원하는 작업을 선택하세요 ======")
    print("1. 연락처 추가")
    print("2. 연락처 삭제")
    print("3. 연락처 수정")
    print("4. 연락처 검색")
    print("5. 연락처 목록 보기")
    print("6. 종료")
    print()

    choice = input("번호 : ")

    print()
    print(choice+"번을 선택하였습니다.")
    print()

    if choice == "1":
        print("연락처를 추가합니다.")
        name = input("이름 : ")
        phone = input("전화번호 : ")
        email = input("이메일 : ")
        address = input("주소 : ")
        add_contact(name, phone, email, address)
    elif choice == "2":
        print("연락처를 삭제합니다.")
        name = input("이름 : ")
        del_contact(name)
    elif choice == "3":
        print("연락처를 수정합니다.")
        name = input("이름 : ")
        mod_contact(name)
    elif choice == "4":
        print("연락처 검색합니다.")
        name = input("이름 : ")
        search_contact(name)
    elif choice == "5":
        print("연락처 목록을 조회합니다.")
        view_contacts()
    elif choice == "6":
        print("종료합니다.")
        break  
    else:
        print("잘못 선택했습니다.")
