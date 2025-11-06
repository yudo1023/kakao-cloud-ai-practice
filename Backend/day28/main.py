from fastapi import FastAPI
from controllers.user_controller import router as user_router
from database import engine
from models import user

user.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title='FastAPI Database 연동 연습',
    description='Repository/Service 패턴을 활용한 FastAPI 애플리케이션',
    version='1.0.0'
)

app.include_router(user_router)

@app.get('/')
def read_root():
    return {
        'message': 'FastAPI Database Integration with Repository/Service Pattern',
        'architecture': 'Layered Architecture',
        'patterns': ['Repository Pattern', 'Service Pattern', 'Dependency Injection']
    }

# 계층 구조 분속
# Client Request -> 
# Controller(HTTP 요청/응답 처리) ->
# Service(비즈니스 로직) ->
# Repository(데이터 액세스) ->
# Database(데이터 저장소)