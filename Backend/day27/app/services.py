from typing import List, Optional
from .models import User, UserCreate, UserUpdate, UserResponse, PaginatedResponse
from .repositories import UserRepository

class UserService:
    def __init__(self, user_reposiroty:UserRepository):
        self.user_repository = user_reposiroty

    def create_user(self, user_data:UserCreate) -> UserResponse:
        existing_user = self.user_repository.get_by_email(user_data.email)
        if existing_user:
            raise ValueError(f"이메일 '{user_data.email}'은 이미 사용중입니당")
        if user_data.age < 14:
            raise ValueError('14세 미만은 가입할 수 없습돵')
        if not user_data.name.strip():
            raise ValueError('이름에 공백은 안되용')
        user = self.user_repository.create(user_data)

        return UserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            age=user.age,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    
    # 사용자 조회
    def get_user(self, user_id:int) -> Optional[UserResponse]:
        user = self.user_repository.get_by_id(user_id)
        if not user:
            return None
        
        return UserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            age=user.age,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    
    # 사용자 목록 조회(페이지네이션)
    def get_users(self, skip:int=0, limit:int=100) -> PaginatedResponse:
        users = self.user_repository.get_all(skip, limit)
        total = self.user_repository.count()

        user_responses = [
            UserResponse(
                id=user.id,
                name=user.name,
                email=user.email,
                age=user.age,
                created_at=user.created_at,
                updated_at=user.updated_at
            )
            for user in users
        ]

        page = (skip // limit) + 1 if limit > 0 else 1
        has_next = skip + limit < total
        has_prev = skip > 0

        return PaginatedResponse(
            items=user_responses,
            total=total,
            page=page,
            size=len(user_responses),
            has_next=has_next,
            has_prev=has_prev
        )
    
    def update_user(self, user_id:int, user_data: UserUpdate) -> Optional[UserResponse]:
        existing_user = self.user_repository.get_by_id(user_id)
        if not existing_user:
            return None
        if user_data.email:
            email_user = self.user_repository.get_by_email(user_data.email)
            if email_user and email_user.id != user_id:
                raise ValueError(f"이메일 '{user_data.email}'은 이미 사용중입니당")
        if user_data.age is not None and user_data.age < 14:
            raise ValueError('14세 미만으로 변경 불가~')
        
        updated_user = self.user_repository.update(user_id, user_data)
        if not updated_user:
            return None
        
        return UserResponse(
            id=updated_user.id,
            name=updated_user.name,
            email=updated_user.email,
            age=updated_user.age,
            created_at=updated_user.created_at,
            updated_at=updated_user.updated_at
        )
    
    def delete_user(self, user_id:int) -> bool:
        return self.user_repository.delete(user_id)