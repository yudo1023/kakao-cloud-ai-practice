from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routers import router as user_router

def create_application() -> FastAPI:
    description="""
    ## FastAPI 실전 프로젝트 구조 예제

    이 API는 다음과 같은 디자인 패턴을 구현합니다:
    - **계층형 아키텍처** : API -> Service -> Repository
    - **Repository 패턴** : 데이터 접근 로직 캡슐화
    - **의존선 주입** : 느슨한 결합과 테스트 용이성
    - **Singleton 패턴** : 효율적인 인스턴스 관리
    
    ### 주요 기능
    - 사용자 CRUD 연산
    - 페이지네이션 지원
    - 데이터 검증 및 에러 처리
    - 통계 정보 제공
    """

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=description,
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        docs_url='/docs',
        redoc_url='/redoc'
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_header=['*'],
    )
    app.include_router(user_router, prefix=settings.API_PREFIX)

    return app

app = create_application()

@app.get('/', tags=['root'])
async def root():
    return {
        'message': f"환영합니다! {settings.PROJECT_NAME}",
        'version': settings.VERSION,
        'docs_url': '/docs',
        'api_prefix': settings.API_PREFIX,
        'environment': settings.ENVIRONMENT
    }

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host='0.0.0.0',
        port=8000,
        reload=settings.DEBUG,
        log_level='info' if not settings.DEBUG else 'debug'
    )