from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List
from .models import UserCreate, UserUpdate, UserResponse, MessageResponse, PaginatedResponse
from .services import UserService
from .dependencies import get_user_service

router = APIRouter(prefix='/users', tags=['users'])

@router.post(
    '/',
    response_model=UserResponse,
    summary='새 사용자 생성',
    description='새로운 사용자를 생성합니다. 이메일 중복 및 나이 제합을 확인합니다.'
)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service)
):
    try:
        return user_service.create_user(user_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='사용자가 생성 중 오류가 발생했습니다'
        )
    
@router.get(
    '/{user_id}',
    response_model=UserResponse,
    summary='사용자 조회',
    description='사용자 ID로 특정 사용자 정보를 조회합니당'
)
async def get_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
):
    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {user_id}인 사용자를 못 찾겟어용"
        )
    return user

@router.get(
    '/',
    response_model=PaginatedResponse,
    summary='사용자 목록 조회',
    description='등록된 사용자 목록을 페이지네이션으로 조회합니당'
)
async def get_users(
    skip: int = Query(0, ge=0, description='건너뛸 사용자 수'),
    limit: int = Query(10, ge=1, le=100, description='조회할 최대 사용자 수'),
    user_service: UserService = Depends(get_user_service)
):
    return user_service.get_users(skip, limit)

@router.put(
    '/{user_id}',
    response_model=UserResponse,
    summary='사용자 정보 수정',
    description='사용자 정보를 수정합니다. 부분 수정을 지원합니당'
)
async def update_user(
    user_id:int,
    user_data: UserUpdate,
    user_service: UserService = Depends(get_user_service)
):
    try:
        update_user = user_service.update_user(user_id, user_data)
        if not update_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ID {user_id}인 사용자 못 찾겠습니당"
            )
        return update_user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='사용자 수정 중 오류가 발생했습니당'
        )
    
@router.delete(
    '/{user_id}',
    response_model=MessageResponse,
    summary='사용자 삭제',
    description='사용자를 삭제합니당(소프트 삭제)'
)
async def delete_user(
    user_id:int,
    user_service: UserService = Depends(get_user_service)
):
    success = user_service.delete_user(user_id)
    if not success:
         raise HTTPException(
             status_code=status.HTTP_404_NOT_FOUND,
             detail=f"ID {user_id}인 사용자를 찾을 수 없어용"
         )
    return MessageResponse(
        message=f"사용자 ID {user_id}기 성공적으로 삭제되었습니당"
    )