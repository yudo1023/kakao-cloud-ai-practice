from sqlalchemy import Column, Integer, String, Float, Text, Boolean
from database import Base

class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    is_available = Column(Boolean, default=True)
    category_id = Column(Integer, nullable=True)