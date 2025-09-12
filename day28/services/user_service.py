from typing import Optional, List
from sqlalchemy.orm import Session
from fastapi import HTTPException
from repositories.user_repository import UserRepository
from schemas.user import UserCreate, UserUpdate, UserResponse
from models.user import User

class UserService:
    def __init__(self, db: Session):
        self.user_repository = UserRepository(db)

    def create_user(self, user_data: UserCreate) -> UserResponse:
        if self.user_repository.get_by_username(user_data.username):
            raise HTTPException(
                status_code=400,
                detail='이미 이름이 있어용'
            )
        db_user = self.user_repository.create(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name
        )

        return UserResponse.from_orm(db_user)
    
    def get_user(self, user_id: int) -> UserResponse:
        user = self.user_repository.get(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail='없는 사용자입니당'
            )
        return UserResponse.from_orm(user)
    
    def get_user(self, skip: int = 0, limit: int = 100) -> List[UserResponse]:
        users = self.user_repository.get_all(skip=skip, limit=limit)
        return [UserResponse.from_orm(user) for user in users]
    
    def update_user(self, user_id: int, user_data: UserUpdate) -> UserResponse:
        existing_user = self.user_repository.get(user_id)
        if not existing_user:
            raise HTTPException(
                status_code=404,
                detail='없는 사용자입니당'
            )
        
        update_data = user_data.dict(exclude_unset=True)

        if 'username' in update_data:
            user_with_username = self.user_repository.get_by_username(update_data['username'])
            if user_with_username and user_with_username.id != user_id:
                raise HTTPException(
                    status_code=400,
                    detail='이미 존재하는 사용자입니당'
                )
            
        updated_user = self.user_repository.update(user_id, **update_data)
        return UserResponse.from_orm(updated_user)
    
    def delete_user(self, user_id: int) -> bool:
        if not self.user_repository.get(user_id):
            raise HTTPException(
                status_code=404,
                detail='없는 사용자입니당'
            )
        return self.user_repository.delete(user_id)
    
    def search_users(self, search_term: str) -> List[UserResponse]:
        users = self.user_repository.search_users(search_term)
        return [UserResponse.from_orm(user) for user in users]
    
       