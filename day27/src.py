from functools import lru_cache
from pydantic_settings import BaseSettings
from fastapi import FastAPI, status, Header, Response, Cookie, UploadFile, File, HTTPException, Form, Depends, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime
import json
import os
import shutil
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from repositories.base_repository import BaseRepository
from repositories.user_repository import UserRepository
# from config import settings
import uvicorn

app = FastAPI()

# # ** FastAPI 기초 **
# # 요청 및 응답 처리 p.171 기본 응답
# @app.get('/api/data')
# async def get_json_data():
#     return {'message': '안녕하세요', 'status': 'success'}

# @app.get('/html', response_class=HTMLResponse)
# async def get_html():
#     return """
#     <html>
#         <head><title>FastAPI HTML</title></head>
#         <body>
#             <h1>안녕하세요, FastAPI!</h1>
#             <p>이것은 HTML 응답입니다.</p>
#         </body>
#     </html>
#     """

# @app.get('/text', response_class=PlainTextResponse)
# async def get_text():
#     return '이것은 단순한 텍스트 응답입니다.'

# @app.get('/download')
# def download_file():
#     file_path = 'example.txt'
#     if not os.path.exists(file_path):
#         with open(file_path, 'w', encoding='utf-8') as f:
#             f.write('하이용~이건 FastAPI에서 생성된 예제 파일입니당\n')
#     return FileResponse(
#         path=file_path,
#         filename='dowloadedfile.text',
#         media_type='text/plain'
#     )

# # p.176 커스텀 응답
# @app.get('/custom_header')
# def custome_header():
#     content = {'message': 'custom header'}
#     response = JSONResponse(content=content)
#     response.headers['X-custom-header'] = 'custom value'
#     response.headers['X-API-Version'] = '1.0'
#     return response

# @app.get('/custom-status')
# def custom_status_response():
#     return JSONResponse(
#         content={'message': 'custom status'},
#         status_code=201
#     )

# @app.get('/error-example')
# def custom_error_response():
#     return JSONResponse(
#         content={'error': '요청한 리소스를 못 찾겠어용'},
#         status_code=404
#     )

# # p.179 http 상태 코드
# @app.get('/success', status_code=status.HTTP_200_OK)
# async def success_response():
#     return {'message': '요청이 성공적으로 처리되었습니다.'}

# @app.post('/create', status_code=status.HTTP_201_CREATED)
# async def create_resource():
#     return {'message': '새 리소스가 생성되었습니다.'}

# @app.delete('/delete/{item_id}', status_code=status.HTTP_204_NO_CONTENT)
# async def delete_resource(item_id:int):
#     return

# @app.get('/bad-request')
# async def bad_request_example():
#     raise HTTPException(
#         status_code=status.HTTP_400_BAD_REQUEST,
#         detail='잘못된 요청입니다.'
#     )

# @app.get("/not-found/{item_id}")
# async def not_found_example(item_id:int):
#     if item_id > 100:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"ID {item_id}인 항목을 찾을 수 없습니다."
#         )
#     return {'item_id': item_id}

# @app.get('/server-error')
# async def server_error_example():
#     raise HTTPException(
#         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         detail='서버 내부 오류가 발생했습니다.'
#     )

# # p.183 http 헤더
# @app.get('/read-headers')
# async def read_headers(
#     user_agent: Optional[str] = Header(None),
#     authorization: Optional[str] = Header(None),
#     custom_header: Optional[str] = Header(None, alias='X-Custom-Header')
# ):
#     return {
#         'user_agent': user_agent,
#         'authorization': authorization,
#         'custom_header': custom_header
#     }

# @app.get('/set-headers')
# async def set_headers(response: Response):
#     response.headers['X-API-Version'] = '1.0'
#     response.headers['X-Rate-Limit'] = '100'
#     response.headers['Cache-Control'] = 'no-cache'
#     return {'message': '헤더가 설정된 응답'}

# @app.get('/cache-forever')
# async def cache_forever(response: Response):
#     response.headers['Cache-Control'] = 'max-age=31535000' # 1년
#     return {'data': '이 데이터는 거의 변하지 않습니다.'}

# @app.get('/cache-short')
# async def cache_short(response: Response):
#     response.headers['Cache-Control'] = 'max-age=300' # 5분
#     return {'timestamp': '2025-09-11 15:05:00', 'data': '자주 업데이트되는 데이터'}

# # p.190 파일 업로드
# @app.post('/upload-file/')
# async def upload_file(file:UploadFile = File(...)):
#     print(f'받은 파일 : {file.filename}')
#     allowed_types = ['image/jpeg', 'image/png']
#     if file.content_type not in allowed_types:
#         raise HTTPException(
#             status_code=400,
#             detail='지원하지 않는 파일 형식입니당~'
#         )
#     content = await file.read()
#     file_size = len(content)
#     if file_size > 1024*1024*5:
#         raise HTTPException(
#             status_code=404,
#             detail='5MB 제한'
#         )
#     upload_dir = 'uploads'
#     os.makedirs(upload_dir, exist_ok=True)

#     file_path = os.path.join(upload_dir, file.filename)
#     with open(file_path, 'wb') as buffer:
#         buffer.write(content)

#     return {
#         'filename': file.filename,
#         'content_type': file.content_type,
#         'size': file_size,
#         'message': '파일이 성공적으로 업로드되었습니다.'
#     }

# # p.195 파일 다운로드
# @app.get('/download/{filename}')
# async def download_file(filename:str):
#     file_path = os.path.join('uploads', filename)

#     if not os.path.exists(file_path):
#         raise HTTPException(
#             status_code=404,
#             detail='파일을 찾을 수 없습니다.'
#         )
#     return FileResponse(
#         path=file_path,
#         filename=filename,
#         media_type='application/octet-stream'
#     )

# # p.196 블로그 API 만들기
# app = FastAPI(title='블로그 API', version='1.0.0')

# class BlogPost(BaseModel):
#     id:int
#     title:str
#     content:str
#     author:str
#     created_at:datetime
#     image_url:Optional[str] = None
#     tags:List[str] = []

# class BlogPostCreate(BaseModel):
#     title:str
#     content:str
#     author:str
#     tags:List[str] = []

# blog_posts = []
# post_id_conter = 1

# @app.get('/posts', response_model=List[BlogPost])
# async def get_posts():
#     return blog_posts

# @app.get('/posts/{post_id}', response_model=BlogPost)
# async def get_post(post_id:int):
#     for post in blog_posts:
#         if post.id == post_id:
#             return post
#     raise HTTPException(
#         status_code=404,
#         detail='블로그 포스트를 찾을 수 없습니다.'
#     )

# @app.post('/posts', response_model=BlogPost, status_code=status.HTTP_201_CREATED)
# async def create_post(
#     tittle:str = Form(...),
#     content:str = Form(...),
#     author:str = Form(...),
#     tags:str = Form(...),
#     image:Optional[UploadFile] = File(None)
# ):
#     global post_id_conter
#     tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
#     image_url = None
#     if image and image.filename:
#         os.makedirs('blog_images', exist_ok=True)
#         image_path = f"blog_images/{post_id_conter}_{image.filename}"
#         content = await image.read()
#         with open(image_path, 'wb') as buffer:
#             buffer.write(content)
#         image_url = f"/images/{post_id_conter}_{image.filename}"
#         new_post = BlogPost(
#             id=post_id_conter,
#             title=tittle,
#             content=content,
#             author=author,
#             created_at=datetime.now(),
#             image_url=image_url,
#             tags=tag_list
#         )
#         blog_posts.append(new_post)
#         post_id_conter += 1

#         return new_post
    
# @app.get('/images/{filename}')
# async def get_image(filename:str):
#     file_path = f'blog_images/{filename}'
#     if not os.path.exists(file_path):
#         raise HTTPException(
#             status_code=404,
#             detail='이미지를 찾을 수 없습니당'
#         )
#     return FileResponse(file_path)

# @app.delete('/posts/{post_id}')
# async def delete_post(post_id:int):
#     global blog_posts
#     for i, post in enumerate(blog_posts):
#         if post.id == post_id:
#             deleted_post = blog_posts.pop(i)
#             if deleted_post.image_url:
#                 filename = deleted_post.image_url.split('/'[-1])
#                 file_path = f'blog_images/{filename}'
#                 if os.path.exists(file_path):
#                     os.remove(file_path)

#             return {'message': '블로그 포스트가 삭제되었습니당'}
#     raise HTTPException(
#         status_code=404,
#         detail='삭제할 블로그 포스트가 없어용'
#     )
         
# # p.210 의존성 함수
# call_count = 0

# def expensive_dependency():
#     global call_count
#     call_count += 1
#     time.sleep(0.1)
#     return {'call count': call_count}

# # Depoends 클래스 캐싱
# @app.get('/test')
# def endpoint1(
#     data1:dict=Depends(expensive_dependency, use_cache=False),
#     data2:dict=Depends(expensive_dependency, use_cache=False),
#     ):
#     return {'test1 data': data1, 'test2 data': data2}

# # 매개변수가 있는 의존성
# def get_query_token(token:str):
#     return token

# @app.get('/items')
# def read_items(token:str = Depends(get_query_token)):
#     return {'token': token}

# # Depends와 타입 힌트
# def get_user_info(user_id:int) -> Dict[str, any]:
#     return {'id': user_id, 'name': '유정', 'email': 'test@example.com'}

# def get_settings() -> Dict[str, any]:
#     return {'debug': True, 'host': 'localhost', 'port': 8000}

# @app.get('/users/{user_id}')
# def get_user(
#     user_id:int,
#     user_info:Dict[str, Any] = Depends(get_user_info),
#     settings:Dict[str, Any] = Depends(get_settings)
# ):
#     return {'user': user_info, 'settings': settings}

# # Depends와 타입 힌트 응용
# class PaginationPrams(BaseModel):
#     page: int
#     size: int

# class UserModel(BaseModel):
#     id: int
#     name: str

# def get_user_agent(user_agent:Optional[str] = Header(None)) -> str:
#     return user_agent

# def get_pagination(page:int=1, size:int=10) -> PaginationPrams:
#     return PaginationPrams(
#         page = max(1, page),
#         size = min(100, max(1, size))
#     )

# def get_current_user(authorization:str = Header()) -> UserModel:
#     if not authorization.startswith('Bearer '):
#         raise HTTPException(
#             status_code=401,
#             detail="유효하지 않는 사용자"
#         )
#     return UserModel(id=1, name='유정', email='test@example.com')

# @app.get('/items')
# def get_items(
#     user_agent = Depends(get_user_agent),
#     pagination:PaginationPrams = Depends(get_pagination),
#     current_user:UserModel = Depends(get_current_user)
# ) -> Dict:
#     return {
#         'user_agent': user_agent,
#         'pagination': pagination.dict(),
#         'current_user': current_user.dict(),
#         'items': ['item1', 'item2', 'item3']
#     }

# # Depends의 고급 사용법
# # 1) 클래스를 의존성으로 사용
# class DatabaseConnection:
#     def __init__(self, host:str='localhost', port:int=5432):
#         self.host = host
#         self.port = port
#         self.connection = f"연결됨~ {host}:{port}"

#     def query(self, sql:str):
#         return f"실행 : {sql} on {self.connection}"
    
# @app.get('/db-query')
# def execute_query(db:DatabaseConnection = Depends(DatabaseConnection)):
#     return db.query('select * from users')

# # 2) HTTP 요청 기반 자동 매개변수 인식 기능 활용
# def get_search_params(
#     q: str = Query(None, description='검색어'),
#     category: str = Query('all', description='카테고리'),
#     sort_by: str = Query('created_at', description='정렬 기준')
# ):
#     return {
#         'query': q,
#         'category': category,
#         'sort_by': sort_by
#     }

# def get_request_info(
#     user_agent: str = Header(None),
#     x_forwarded_for: str = Header(None)
# ):
#     return {
#         'user_agent': user_agent,
#         'client_ip': x_forwarded_for
#     }

# @app.get('/search')
# def search_items(
#     search_params: dict = Depends(get_search_params),
#     request_info: dict = Depends(get_request_info)
# ):
#     return {
#         'search_params': search_params,
#         'request_info': request_info,
#         'results': ['item1', 'item2']
#     }

# # p.232 설정 의존성
# class Settings(BaseSettings):
#     database_url:str = 'sqlite:///./test.db'
#     sercret_key:str = 'your-secret-key-here'
#     api_key:Optional[str] = None

#     class Config:
#         env_file = '.env'
#         env_file_encoding = 'utf-8'

# @lru_cache
# def get_settings() -> Settings:
#     return Settings()

# @app.get('/info')
# def get_app_info(settings:Settings = Depends(get_settings)):
#     return {
#         'database_url': settings.database_url,
#         'secret_key': settings.sercret_key,
#         'api_key': settings.api_key
#     }

# # p.235 의존성 체인
# def get_token_header(x_token:str = Header()):
#     return x_token

# def get_current_user(token: str = Depends(get_token_header)):
#     if token != 'fake-super-secret-token':
#         raise HTTPException(
#             status_code=400,
#             detail='X-Token header invalid'
#         )
#     return {'username': 'authenticated_user'}

# @app.get('/protected-route')
# def protected_route(current_user: dict = Depends(get_current_user)):
#     return {'message': f"안뇽 {current_user['username']}"}

# # ** FastAPI 디자인 패턴 **
# # p.12 Repository 패턴
# # /repositories/base_repository.py
# # /repositories/user_repository.py

# # p.21 싱글턴 패턴
# @lru_cache
# def get_user_repository() -> UserRepository:
#     return UserRepository()

# def get_user_service(
#         user_repository:UserRepository = Depends(get_user_repository)
# ):
#     return {'message': 'user service initalized'}

# # p.24 factory 패턴
# # /factories/service_factory.py

# # observe 패턴
# @dataclass
# class Event:
#     name:str
#     data:Any
#     timestamp:datetime = datetime.now()

# class EventManager:
#     def __init__(self):
#         self._observers:dict[str, List[Callable]] = {}
    
#     def subscribe(self, event_name:str, callback:Callable):
#         # 이벤트 구독
#         if event_name not in self._observers:
#             self._observers[event_name] = []
#         self._observers[event_name].append(callback)
    
#     def publish(self, event:Event):
#         if event.name in self._observers:
#             for callback in self._observers[event.name]:
#                 try:
#                     callback(event)
#                 except Exception as e:
#                     print(f"{e}")
        
# # 이벤트 핸들러
# def send_welcome_email(event:Event):
#     user_data = event.data
#     print(f"환영 이메일 발송: {user_data['email']}")

# def log_user_creation(event:Event):
#     user_data = event.data
#     print(f"사용자 생성 로그: {user_data['name']} at {event.timestamp}")

# event_manager = EventManager()
# # 이벤트 구독
# event_manager.subscribe('user_created', send_welcome_email)
# event_manager.subscribe('user_created', log_user_creation)

# # 유저가 생성이 되면 
# event_manager.publish(Event(name='user_created', data={'name': '유정', 'email': 'test@example.com'}))

# p.34 실전 프로젝트 구조 구현
# 1) 환경변수 파일 작성 : /app/.env
# 2) 모듈 초기화 파일 작성 : /app/__init__.py
# 3) 설정 파일 구성 : /app/config.py
# 4) 데이터 모델 정의 : /app/models.py
# 5) 데이터 접근 계층 : /app/respositories.py