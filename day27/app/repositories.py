from typing import List, Optional
from datetime import datetime
import hashlib
from .models import User, UserCreate, UserUpdate

class UserRepository:
    def __init__(self):
        self._users:List[User] = []
        self._next_id = 1
        self._add_sample_data()

    def _add_sample_data(self):
        sample_users = [
            UserCreate(
                name='곽유정',
                email='yu@example.com',
                age=30,
                password='password123'
            ),
            UserCreate(
                name='강윤석',
                email='yun@example.com',
                age=30,
                password='password123'
            )
        ]
        
        for user_data in sample_users:
            self.create(user_data)
    
    def _hash_password(self, password:str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create(self, user_data:UserCreate) -> User:
        user = User(
            id=self._next_id,
            name=user_data.name,
            email=user_data.email,
            age=user_data.age,
            password_hash=self._hash_password(user_data.password),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self._users.append(user)
        self._next_id += 1
        return user
    
    def get_by_id(self, user_id: int) -> Optional[User]:
        for user in self._users:
            if user.id == user_id and user.is_active:
                return user
        return None
    
    def get_by_email(self, email:str) -> Optional[User]:
        for user in self._users:
            if user.email == email and user.is_active:
                return user
        return None
    
    def get_all(self, skip:int=0, limit:int=100) -> List[User]:
        active_users = [user for user in self._users if user.is_active]
        return active_users[skip:skip+limit]
    
    def update(self, user_id:int, user_data:UserUpdate) -> Optional[User]:
        user = self.get_by_id(user_id)
        if not user:
            return None
        update_data = user_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        user.updated_at = datetime.now()
        return user
    
    def delete(self, user_id:int) -> bool:
        user = self.get_by_id(user_id)
        if user:
            user.is_active = False
            user.updated_at = datetime.now()
            return True
        return False
    
    def count(self) -> int:
        return len([user for user in self._users if user.is_active])