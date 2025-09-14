# p.236
# 1. 목표 : 전자상거래 API 의존성 시스템 구현
# 요구사항
# 사용자 인증 의존성 구현 - Bearer Token으로 사용자 확인
# 관리자 권한 확인 의존성 구현 - 관리자만 접근 가능한 기능

from datetime import datetime
from typing import List, Optional
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

app = FastAPI(title="전자상거래 API 의존성 시스템", version="1.0.0")

class User(BaseModel):
    username: str
    full_name: Optional[str] = None
    roles: List[str] = []
    disabled: bool = False

TOKENS: dict[str, User] = {
    "user-token-123": User(username="alice", full_name="Alice Customer", roles=["customer"], disabled=False),
    "admin-token-abc": User(username="bob", full_name="Bob Admin", roles=["admin"], disabled=False),
}

bearer_scheme = HTTPBearer(auto_error=True)

# 인증
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> User:
    token = credentials.credentials
    user = TOKENS.get(token)
    if not user:
        # 인증 실패
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증 정보가 올바르지 않습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

def admin_required(user: User = Depends(get_current_user)) -> User:
    if "admin" not in (user.roles or []):
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    return user

@app.get("/mypage", summary="내 정보 조회")
async def read_me(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "full_name": current_user.full_name,
        "roles": current_user.roles,
    }

@app.get("/orders", summary="내 주문 목록")
async def list_my_orders(current_user: User = Depends(get_current_user)):
    now = datetime.now().isoformat()
    return [
        {"order_id": 101, "user": current_user.username, "status": "paid", "at": now},
        {"order_id": 102, "user": current_user.username, "status": "shipped", "at": now},
    ]

@app.post("/admin/products", summary="상품 등록(관리자)")
async def create_product_admin(_admin: User = Depends(admin_required)):
    return {"message": "product created", "by": _admin.username, "time": datetime.now().isoformat()}

@app.delete("/admin/users/{username}", summary="사용자 삭제(관리자)")
async def delete_user_admin(username: str, _admin: User = Depends(admin_required)):
    return {"deleted": username, "by": _admin.username}

@app.get("/")
async def root():
    return {
        "app": "전자상거래 API 의존성 시스템",
        "version": "1.0.0",
        "docs": "/docs",
        "notes": [
            "Authorization: Bearer <token> 헤더 필요",
            "샘플 토큰: user-token-123 (일반), admin-token-abc (관리자)",
        ],
        "endpoints": [
            {"GET": "/mypage"},
            {"GET": "/orders"},
            {"POST": "/admin/products"},
            {"DELETE": "/admin/users/{username}"},
        ],
    }
