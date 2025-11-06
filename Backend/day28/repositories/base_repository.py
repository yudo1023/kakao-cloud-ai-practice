from typing import Generic, Type, TypeVar, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import DeclarativeMeta

ModelType = TypeVar('MdoelType', bound=DeclarativeMeta)

class BaseRepository(Generic[ModelType]):
    def __init__(self, model:Type[ModelType], db:Session):
        self.model = model
        self.db = db
    
    # id로 단일 객체 조회
    def get(self, id:int) -> Optional[ModelType]:
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    # 모든 객체 조회(페이징)
    def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        return self.db.query(self.model).offset(skip).limit(limit).all()
    
    # 새 객체 생성
    def create(self, **kwargs) -> ModelType:
        db_obj = self.model(**kwargs)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    # 객체 수정
    def update(self, id: int, **kwargs) -> Optional[ModelType]:
        db_obj = self.get(id)
        if db_obj:
            for key, value in kwargs.items():
                if hasattr(db_obj, key):
                    setattr(db_obj, key, value)
            self.db.commit()
            self.db.refresh(db_obj)
        return db_obj
    
    # 객체 삭제
    def delete(self, id: int) -> bool:
        db_obj = self.get(id)
        if db_obj:
            self.db.delete(db_obj)
            self.db.commit()
            return True
        return False