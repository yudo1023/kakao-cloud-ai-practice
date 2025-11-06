from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    name:str = Field(..., min_length=2, max_length=50, description="이름")
    email:EmailStr = Field(..., description="이메일 주소")
    age: int = Field(..., ge=0, le=150, description="나이(0-150)")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="비밀번호(최소 8자)")

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=50)
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, ge=0, le=150)

class UserResponse(UserBase):
    id: int = Field(..., description='사용자 고유 ID')
    created_at: datetime = Field(..., description='생성 일시')
    updated_at: datetime = Field(..., description='수정 일시')

class User(UserBase):
    id: Optional[int] = None
    password_hash: str = Field(..., description='해시된 비밀번호')
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(default=True, description='활성 상태')

class MessageResponse(BaseModel):
    message: str
    success: bool = True

class ErrorResponse(BaseModel):
    message: str
    error_code: Optional[str] = None
    success: bool = False

class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    size: int
    has_next: bool
    has_prev: bool