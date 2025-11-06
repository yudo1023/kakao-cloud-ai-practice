from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from services.user_service import UserService
from schemas.user import UserCreate, UserUpdate, UserResponse

router = APIRouter(prefix='/users', tags=['users'])

def get_user_service(db: Session = Depends(get_db)) -> UserService:
    return UserService(db)

@router.post('/', response_model=UserResponse, status_code=201)
def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service)
):
    return user_service.create_user(user_data)

@router.get('/', response_model=List[UserResponse])
def get_users(
    skip: int = 0,
    limit: int = 100,
    user_service: UserService = Depends(get_user_service)
):
    return user_service.get_user(skip=skip, limit=limit)

@router.get('/{user_id}', response_model=UserResponse)
def get_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
):
    return user_service.get_user(user_id)

@router.put('/{user_id}', response_model=UserResponse)
def update_user(
    user_id: int,
    user_data: UserUpdate,
    user_service: UserService = Depends(get_user_service)
):
    return user_service.update_user(user_id, user_data)

@router.delete('/{user_id}')
def delete_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
):
    success = user_service.delete_user(user_id)
    if success:
        return {'message': '사용자 삭제가 성공적으로 되었습니당'}
    raise HTTPException(status_code=400, detail='사용자 삭제 실패~')