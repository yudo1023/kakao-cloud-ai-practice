from functools import lru_cache
from .repositories import UserRepository
from .services import UserService
from fastapi import Depends

@lru_cache()
def get_user_repository() -> UserRepository:
    return UserRepository()

def get_user_service(
    user_repository:UserRepository = Depends(get_user_repository)
) -> UserService:
    return UserService(user_repository)