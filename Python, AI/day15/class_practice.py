# p.131
# 1. 목표 : 다음 클래스를 구현하세요
# - Book: 도서 정보(제목, 저자, ISBN, 출판연도 등)를 관리
# - Library: 도서 컬렉션을 관리하고 대출/반납 기능 제공
# - Member: 도서관 회원 정보와 대출 목록 관리
# 2. 요구 사항
# 도서 추가/삭제, 도서 검색(제목, 저자, ISBN으로), 도서 대출/반납, 회원 등록/관리, 회원별 대출 현황 확인
# 객체 지향 설계 원칙(SOLID)을 최소한 2가지 이상 적용하세요.
# 적절한 캡슐화를 통해 데이터를 보호하세요.

# Book 클래스
class Book:
    def __init__(self, title, author, isbn, publish_year):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.publish_year = publish_year
        self.is_borrowed = False

    def info(self):
        return f"[도서] 제목: {self.title}, 저자: {self.author}, ISBN: {self.isbn}, 출판연도: {self.publish_year}, 대출여부: {self.is_borrowed}"

    def show(self):
        print(self.info())

# Member 클래스
class Member:
    _seq = 0

    def __init__(self, name):
        Member._seq += 1
        self.member_id = f"M{Member._seq:04d}"
        self.name = name
        self.borrowed_books = []

    def info(self):
        titles = [b.title for b in self.borrowed_books]
        return f"[회원] ID: {self.member_id}, 이름: {self.name}, 대출중: {titles}"

    def show(self):
        print(self.info())

# Library 클래스
class Library:
    def __init__(self):
        self._books = []
        self._members = {}

    # 도서 추가
    def add_book(self, book):
        if self.get_book_by_isbn(book.isbn):
            print("동일한 ISBN의 도서가 존재합니다.")
            return False
    
        self._books.append(book)
        print(f"[도서 추가] 도서명 : {book.title}")
        return True
    
    # 도서 삭제
    def remove_book(self, isbn):
        book = self.get_book_by_isbn(isbn)
        if not book:
            print("해당 ISBN의 도서가 없습니다.")
            return False
        
        if book.is_borrowed:
            print("대출 중인 도서는 삭제할 수 없습니다.")
            return False
        
        self._books.remove(book)
        print(f"[도서 삭제] 도서명 : {book.title} 도서가 삭제되었습니다.")
        return True
    
    # isbn 도서 검색
    def get_book_by_isbn(self, isbn):
        for book in self._books:
            if book.isbn == isbn:
                return book
        return None
    
    # 도서 검색
    def search_by_title(self, keyword: str):
        key = keyword.lower()
        return [b for b in self._books if key in b.title.lower()]

    def search_by_author(self, keyword: str):
        key = keyword.lower()
        return [b for b in self._books if key in b.author.lower()]

    def search_by_isbn(self, keyword: str):
        key = keyword.lower()
        return [b for b in self._books if key in b.isbn.lower()]
    
     # 회원 등록
    def add_member(self, user_name):
        member = Member(user_name)
        self._members[member.member_id] = member
        print(f"[회원 등록] 회원명 : {member.name}, ID: {member.member_id} 회원이 등록되었습니다.")
        return member
    
    # 회원 삭제
    def remove_member(self, member_id):
        member = self._members.get(member_id)
        if not member:
            print("해당 회원이 존재하지 않습니다.")
            return False
        
        if member.borrowed_books:
            print("대출 중인 도서가 있어 회원 삭제가 불가합니다.")
            return False
        
        del self._members[member_id]
        print(f"[회원 삭제] 회원명 : {member.name}, ID: {member.member_id} 회원이 삭제되었습니다.")
        return True
    
    def get_member(self, member_id):
        return self._members.get(member_id)
    
    # 회원별 대출 현황 확인
    def borrow_list(self, member_id):
        member = self.get_member(member_id)

        if not member:
            print("존재하지 않는 회원입니다.")
            return []
        
        return list(member.borrowed_books)
    
    # 도서 대출
    def borrow_book(self, member_id, isbn):
        member = self.get_member(member_id)

        if not member:
            print("존재하지 않는 회원입니다.")
            return False
        
        book = self.get_book_by_isbn(isbn)

        if not book:
            print("존재하지 않는 도서입니다.")
            return False
        
        if book.is_borrowed:
            print(f"'{book.title}' 은(는) 이미 대출 중입니다.")
            return False

        book.is_borrowed = True
        member.borrowed_books.append(book)
        print(f"[도서 대출] {member.name} 님이 '{book.title}' 을(를) 대출했습니다.")
        return True

    # 도서 반납
    def return_book(self, member_id, isbn):
        member = self.get_member(member_id)

        if not member:
            print("존재하지 않는 회원입니다.")
            return False
        
        book = self.get_book_by_isbn(isbn)

        if not book:
            print("존재하지 않는 도서입니다.")
            return False
        
        if book not in member.borrowed_books:
            print(f"{member.name} 님은 '{book.title}' 을(를) 빌린 적이 없습니다.")
            return False

        book.is_borrowed = False
        member.borrowed_books.remove(book)
        print(f"[도서 반납] {member.name} 님이 '{book.title}' 을(를) 반납했습니다.")
        return True
    
    # 도서 목록
    def list_books(self):
        return list(self._books)
    
    # 회원 목록
    def list_members(self):
        return list(self._members.values())
    
# -- 콘솔 출력용 --
def print_books(books):
    if not books:
        print("도서가 없습니다.")
        return
        
    for book in books:
        print(book.info())
    
def print_members(members):
    if not members:
        print("회원이 없습니다.")
        return
        
    for member in members:
        print(member.info())

def handle_add_book(lib):
    title = input("제목 : ").strip()
    author = input("저자 : ").strip()
    isbn = input("ISBN : ").strip()
    year = input("출판연도(숫자) : ").strip()
    try:
        year = int(year)
    except ValueError:
        print("출판연도는 숫자여야 합니다.")
        return
    lib.add_book(Book(title, author, isbn, year))

def handle_remove_book(lib):
    isbn = input("삭제할 ISBN : ").strip()
    lib.remove_book(isbn)

def handle_search_book(lib):
    print("검색 기준을 선택하세요: 1) 제목  2) 저자  3) ISBN")
    t = input("번호 : ").strip()
    keyword = input("검색어 : ").strip()
    print()

    if t == "1":
        results = lib.search_by_title(keyword)
    elif t == "2":
        results = lib.search_by_author(keyword)
    elif t == "3":
        results = lib.search_by_isbn(keyword)
    else:
        print("잘못 선택하셨습니다.")
        return

    if not results:
        print("검색 결과가 없습니다.")
    else:
        print(f"검색 결과({len(results)}건) :")
        print_books(results)

def handle_add_member(lib):
    name = input("회원 이름 : ").strip()
    lib.add_member(name)

def handle_remove_member(lib):
    member_id = input("삭제할 회원 ID : ").strip()
    lib.remove_member(member_id)

def handle_borrow(lib):
    member_id = input("회원 ID : ").strip()
    isbn = input("대출할 도서의 ISBN : ").strip()
    lib.borrow_book(member_id, isbn)

def handle_return(lib):
    member_id = input("회원 ID : ").strip()
    isbn = input("반납할 도서의 ISBN : ").strip()
    lib.return_book(member_id, isbn)

def handle_borrow_list(lib):
    member_id = input("조회할 회원 ID : ").strip()
    books = lib.borrow_list(member_id)
    if books:
        print(f"[{member_id}] 대출 현황 :")
        print_books(books)

lib = Library()

book1 = Book("book1", "author1", "1111", 2025)
book2 = Book("book2", "author2", "2222", 2020)

lib.add_book(book1)
lib.add_book(book2)

memeber1 = lib.add_member("홍길동")

while True:
    print()
    print("====== 원하는 작업을 선택하세요 ======")
    print("1. 도서 추가")
    print("2. 도서 삭제")
    print("3. 도서 검색")
    print("4. 회원 등록")
    print("5. 회원 삭제")
    print("6. 도서 대출")
    print("7. 도서 반납")
    print("8. 회원별 대출 현황")
    print("9. 전체 도서 목록 보기")
    print("10. 전체 회원 목록 보기")
    print("0. 종료")
    print()

    choice = input("번호 : ").strip()

    print()
    print(choice+"번을 선택하였습니다.")
    print()


    if choice == "1":
        handle_add_book(lib)
    elif choice == "2":
        handle_remove_book(lib)
    elif choice == "3":
        handle_search_book(lib)
    elif choice == "4":
        handle_add_member(lib)
    elif choice == "5":
        handle_remove_member(lib)
    elif choice == "6":
        handle_borrow(lib)
    elif choice == "7":
        handle_return(lib)
    elif choice == "8":
        handle_borrow_list(lib)
    elif choice == "9":
        print_books(lib.list_books())
    elif choice == "10":
        print_members(lib.list_members())
    elif choice == "0":
        print("종료합니다.")
        break
    else:
        print("잘못 누르셨습니다.")