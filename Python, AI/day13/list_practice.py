# p.105
# 1. 목표 : 학생들의 이름과 점수 정보를 리스트로 관리하는 코드 구현
# 2. 요구사항
# 학생 추가 : 이름과 점수를 입력 받아 목록에 추가
# 학생 삭제 : 이름을 입력 받아 해당 학생 정보 삭제
# 성적 수정 : 이름을 입력 받아 해당 학생의 점수 수정
# 전체 목록 출력 : 모든 학생의 이름과 점수 출력
# 통계 출력 : 최고 점수, 최저 점수, 평균 점수 계산 및 출력

students = [["홍길동", 95], ["김철수", 80]]

# 학생 추가
def add_student(name, score):
    students.append([name,score])
    print(f"학생 '{name}'이(가) 추가되었습니다.")

# 학생 삭제
def del_student(name):
    for student in students:
        if student[0] == name:
            students.remove(student)
            print(f"학생 {name}이(가) 삭제되었습니다.")
            return  
    print("해당 학생이 없습니다.")

# 성적 수정
def mod_student(name):
    for student in students:
        if student[0] == name:
            print("수정할 항목을 선택하세요 : (1) 성적 (2) 취소")
            mod_choice = input("번호 : ")

            if mod_choice == "1":
                print(f"현재 점수 : {student[1]}")
                new_score = int(input("새 점수 : "))
                student[1] = new_score
                print("수정했습니다.")
                return
            elif mod_choice == "2":
                print("취소하였습니다.")
                return
            else:
                print("잘못 선택했습니다.")
                return

    print("해당 학생이 없습니다.")

# 학생 목록 보기
def view_student():
    if not students:
        print("학생 정보가 없습니다.")
    else:
        print("*** 학생 목록 ***")
        for student in students:
            print(f"이름 : {student[0]}, 점수 : {student[1]}")
        print()

# 성적 통계 보기
def statistic_student():
    if not students:
        print("학생 정보가 없습니다.")
    else:
        scores = [s[1] for s in students]
        # 최고 점수
        max_score = max(scores)
        # 최저 점수
        min_score = min(scores)
        # 평균 점수
        avg_score = sum(scores)/len(scores)

        print("*** 성적 통계 ***")
        print(f"최고 점수 : {max_score}")
        print(f"최저 점수 : {min_score}")
        print(f"평균 점수 : {avg_score:.2f}")

while True:
    print()
    print("====== 원하는 작업을 선택하세요 ======")
    print("1. 학생 추가")
    print("2. 학생 삭제")
    print("3. 성적 수정")
    print("4. 학생 목록 보기")
    print("5. 성적 통계 보기")
    print("6. 종료")
    print()

    choice = input("번호 : ")

    print()
    print(choice+"번을 선택하였습니다.")
    print()

    if choice == "1":
        print("학생을 추가합니다.")
        name = input("이름 : ")
        score = int(input("점수 : "))
        add_student(name, score)
    elif choice == "2":
        print("학생을 삭제합니다.")
        name = input("이름 : ")
        del_student(name)
    elif choice == "3":
        print("성적을 수정합니다.")
        name = input("이름 : ")
        mod_student(name)
    elif choice == "4":
        print("학생 목록을 조회합니다.")
        view_student()
    elif choice == "5":
        print("성적 통계 조회합니다.")
        statistic_student()
    elif choice == "6":
        print("종료합니다.")
        break
    else:
        print("잘못 선택했습니다.")

