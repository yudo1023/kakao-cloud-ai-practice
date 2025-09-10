from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, Query, status, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ValidationError
import re
from typing import Optional, List
import uvicorn
from routers import users, products
from datetime import datetime

# @app.get('/')
# def read_root():
#     return {'message': 'FastAPI 호출 완료'}

# @app.get('/hello')
# def read_root():
#     return {'message': 'Hello World'}

# @app.post('/order')
# def create_order():
#     return {'message': 'order created'}

# @app.get('/users/{user_id}')
# def get_user(user_id:int):
#     return {'user_id': user_id}

# @app.get('/cafe/{menu_type}/{item_name}')
# def get_menu_item(menu_type:str, item_name:str):
#     return {'category': menu_type,
#             'item_name': item_name}

# if __name__ == "__main__":
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# # ----------------------
# # p.64 Pydantic 응답 모델
# class Order(BaseModel):
#     order_id:int
#     customer_name:str
#     total_amount:float

# class OrderList(BaseModel):
#     orders:List[Order]
#     total_count:int

    
# app = FastAPI()
# @app.get('/orders', response_model=OrderList)
# def get_orders():
#     return {
#         'orders':[
#             {'order_id': 1, 'customer_name': '김철수', 'total_amount': 15000}
#         ],
#         'total_count': 1
#     }

# @app.post('/orders', response_model=Order, status_code=201)
# def create_order(order_data:dict):
#     return {'order_id': 123, 'customer_name': '이영희', 'total_amount': 25000}

# class Order(BaseModel):
#     menu_item:str
#     quantity:int
#     customer_name:str

# @app.post('/orders')
# def create_order(order:Order):
#     return {
#         'message':'접수',
#         'menu': order.menu_item
#     }

# @app.get('/restaurant/order')
# def order_food(
#     age:int = Query(..., ge=18, le=65, description='주문자 나이'),
#     people:int = Query(2, gt=0, lt=11, description='식사 인원'),
#     budget:float = Query(None, ge=100000, description='예산')
# ):
#     return {'주문정보':f"나이: {age}, 인원: {people}, 예산: {budget}"}

# @app.get('/products')
# def get_products(
#     category:Optional[str] = None, # optional: 필수적으로 오지 않는 경우
#     min_price:Optional[float] = None,
#     max_price:Optional[float] = None,
#     tags:List[str] = Query(default=[]), # 몇개가 올지 모를땐 list로
# ):
#     filters = {
#         'category':category,
#         'price_range':f"{min_price} ~ {max_price}",
#         'tags':tags
#     }
#     return {'filters':filters}

# # ----------------------
# # p.79 라우터 그룹화 및 관리
# app = FastAPI(
#     title="My Store API",
#     description="온라인 상점을 위한 RESTful API",
#     version="1.0.0",
# )

# app.include_router(users.router)
# app.include_router(products.router)

# # ----------------------
# # p.110 상세 API 문서 작성
# app = FastAPI(
#     title='도서관 관리 시스템',
#     description="""
#     도서관 관리 시스템 API
#     API 설명 상세 설명
#     """,
#     version='1.0.0',
#     contact={
#         "name": "개발팀",
#         "email": "dev@example.com",
#     }
# )

# class Config:
#     schema_extra = {
#         "example": {
#             "title": "FastAPI 완벽 가이드",
#             "author": "이개발자",
#             "published_data": "2025-09-10T16:43:00"
#         }
#     }

# class Book(BaseModel):
#     title:str=Field(..., description='도서 제목', examples=['파이썬 FastAPI']),
#     author:str=Field(..., description='저자', examples=['유정']),
#     published_date:Optional[datetime]=Field(None, description='출판일')

# class BookResponse(Book):
#     id:int
#     title: str
#     author: str
#     status: str='등록됨'
#     created_at: datetime

# @app.post(
#     '/books',
#     response_model=BookResponse,
#     status_code=status.HTTP_201_CREATED
# )
# def create_book(book: Book):
#     new_book = BookResponse(
#         id=1,
#         title=book.title,
#         author=book.author,
#         created_at=datetime.now()
#     )
#     return new_book

# @app.get(
#         '/books/{book_id}',
#         description="특정 도서 조회",

# )
# def get_book(book_id:int):
#     if book_id == 999:
#         raise HTTPException(
#             status_code=404,
#             detail='book not found'
#         )
#     return BookResponse(
#         id=book_id,
#         title="FASTAPI 가이드",
#         author="개발자",
#         created_at=datetime.now()
#     )

# # ----------------------
# # p.122 태그 작성
# tags_metadata = [
#     {
#         'name': '도서 관리',
#         'description': '도서의 등록, 조회, 수정, 삭제 기능을 제공합니다.',
#         'externalDocs': {
#             'description': '도서 관리 매뉴얼',
#             'url': 'https://example.com/docs/books',
#         },
#     },
#     {
#         'name': '사용자 관리',
#         'description': '회원 가입, 로그인, 정보 수정 기능을 제공합니다.',
#     },
#     {
#         'name': '대출 관리',
#         'description': '도서 대출 및 반납 관리 기능을 제공합니다.',
#     },
# ]

# app = FastAPI(
#     title='도서관 관리 시스템',
#     description='도서관 운영을 위한 종합 관리 시스템',
#     version='1.0.0',
#     openapi_tags=tags_metadata
# )

# @app.get('/books', tags=['도서 관리'])
# def get_books():
#     return [{"id": 1, "title": "FastAPI 가이드"}]

# @app.post('/books', tags=['도서 관리'])
# def create_book():
#     return {'messsage': '도서가 등록되었습니다'}

# @app.get('/users', tags=['사용자 관리'])
# def get_users():
#     return [{'id': 1, 'name': '사용자1'}]

# @app.post('/users/register', tags=['사용자 관리'])
# def register_user():
#     return {'message': '회원가입이 완료되었습니다'}

# @app.post('/loans', tags=['대출 관리'])
# def create_loan():
#     return {'message': '대출이 완료되었습니다.'}

# @app.put('/loan/{item_id}/return', tags=['대출 관리'])
# def return_book(loan_id:int):
#     return {'message': f'대출 id {loan_id} 반납 완료'}


# app = FastAPI(
#     title='My Store API',
#     description='온라인 몰 REST API',
#     version='1.0.0'
# )

# app.include_router(users.router)
# app.include_router(products.router)

# @app.get('/items')
# def read_time(skip:int=0, limit:int=10):
#     items = []
#     for i in range(skip,skip + limit):
#         items.append({
#             "id": i,
#             "name": f"Item {i}",
#             "description": f"Description for item {i}"
#         })
#     return items

# # ----------------------
# # p.131 FastAPI 활용 Bearer Token 적용
# security = HTTPBearer()
# app = FastAPI()

# @app.get('/users/me', dependencies=[Depends(security)])
# def get_current_user(credentials:HTTPAuthorizationCredentials = Depends(security)):
#     token = credentials.credentials
#     print(token)
#     return {'message': token}

# # ----------------------
# # p.147 커스텀 검증자
# class UserRegisteration(BaseModel):
#     username:str
#     password:str
#     confirm_password:str
#     email:str
#     age:int

#     @field_validator('username')
#     @classmethod
#     def username_must_be_alphanumeric(cls, v):
#         if not re.match(f'^[a-zA-Z0-9]+$', v):
#             raise ValueError('사용자명은 영문자와 숫자만 사용 가능합니다.')
#         return v


#     @field_validator('password')
#     @classmethod
#     def password_stregnth(cls, v):
#         if len(v) < 8:
#             raise ValueError('비밀번호는 최소 8자 이상이어야 합니다.')
#         if not re.search(r'[A-Z]',v):
#             raise ValueError('비밀번호에 대문자 필수 포함!!')
#         if not re.search(r'[0-9]', v):
#             raise ValueError('비밀번호에 숫자 필수!!')
#         return v
    
#     @field_validator('confirm_password')
#     @classmethod
#     def password_match(cls, v, info):
#         if info.data and 'password' in info.data and v != info.data['password']:
#             raise ValueError('비밀번호 일치하지 않음')
#         return v
    
#     @field_validator('email')
#     @classmethod
#     def email_must_be_valid(cls, v):
#         if not re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$', v):
#             raise ValueError('유효한 이메일 형식이 아닙니다.')
#         return v

#     @field_validator('age')
#     @classmethod
#     def age_must_be_adult(cls, v):
#         if v < 18:
#             raise ValueError('18세 이상만 가입 가능')
#         return v
    
# try:
#     user = UserRegisteration(
#         username='john123',
#         password='john123',
#         confirm_password='john123',
#         email='john@example.com',
#         age=25
        
#     )
#     print('회원가입 성공')
# except ValidationError as e:
#     print(f"검증 오류: {e}")

# # ----------------------
# # p.152 검증 에러
# class Student(BaseModel):
#     name: str
#     age: int
#     grades: List[float]
#     email: str

# try:
#     student = Student(
#         name="",
#         age="not_a_number",
#         grades=['A', 'B'],
#         email="invalid_email"
#     )
# except ValidationError as e:
#     print('검증 에러 발생:')
#     for error in e.errors():
#         print(f"- 필드: {error['loc']}")
#         print(f"- 메시지: {error['msg']}")
#         print(f"- 타입: {error['type']}")
#         print(f"- 입력값: {error['input']}")
#         print()
