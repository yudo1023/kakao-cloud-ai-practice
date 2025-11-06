from typing import List, Optional
from base_repository import BasecRepository

class User:
    id:int
    name:str
    email:str
    age:int

class UserCreate:
    name:str
    email:str
    age:int

class UserRepository(BasecRepository[User]):
    def __init__(self):
        self.users = []
        self.next_id = 1

    def create(self, user_data:UserCreate) -> User:
        user = User(
            id=self.next_id,
            name=user_data.name,
            email=user_data.email,
            age=user_data.age
        )
        self.users.append(user)
        self.next_id += 1
        return user
    
    def get_by_id(self, user_id:int) -> Optional[User]:
        for user in self.users:
            if user_id == user_id:
                return user
        return None
       