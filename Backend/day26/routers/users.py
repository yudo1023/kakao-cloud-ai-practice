from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(
    prefix='/users',
    tags=['users'],
    responses={404: {"description": "Not Found"}}
)

class User(BaseModel):
    id:int
    name:str
    email:str
    is_active:bool=True
    is_del:bool=False # soft delete
    
class UserCreate(BaseModel):
    name:str
    email:str

fake_users_db = [
    User(id=1, name="Alice", email="alice@example.com"),
    User(id=2, name="Bob", email="bob@example.com"),
]

# CRUD
@router.get('/', response_model=List[User])
def read_users(skip:int=0, limit:int=100):
    return fake_users_db[skip:skip+limit]

@router.post('/', response_model=User)
def create_user(user:UserCreate):
    new_id = len(fake_users_db)+1
    new_user = User(id=new_id, **user.dict())
    fake_users_db.append(new_user)
    return new_user

@router.get("/{user_id}", response_model=User)
def read_user(user_id:int):
    for user in fake_users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail='user not found')

@router.put("/{user_id}", response_model=User)
def update_user(user_id:int, user_update:UserCreate):
    for i, user in enumerate(fake_users_db):
        if user.id == user_id:
            updated_user = User(id=user_id, **user_update.dict())
            fake_users_db[i] = updated_user
            return update_user
    raise HTTPException(status_code=404, detail="user not found")
    
@router.delete('/{user_id}')
def delete_user(user_id:int):
    for i, user in enumerate(fake_users_db):
        if user.id == user_id:
            del fake_users_db[i]
            return {'message': f"user {user_id} delete successfully"}
    raise HTTPException(status_code=404, detail='user not found')