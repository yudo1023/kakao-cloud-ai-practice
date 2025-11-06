from enum import Enum
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(
    prefix='/products',
    tags=['products']
)

class CategoryEnum(str, Enum):
    electronics = "electronics"
    clothing = "clothing"
    books = "books"

class Product(BaseModel):
    id: int
    name: str
    price: float
    category: CategoryEnum
    description: Optional[str] = None

fake_products_db = [
    Product(id=1, name="iphone17pro", price=2000000, category="electronics"),
    Product(id=2, name="shirts", price=10000, category="clothing"),
    Product(id=3, name="python book", price=20000, category="books"),
]

@router.get("/", response_model=List[Product])
def read_products(
    category: Optional[CategoryEnum] = None,
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    skip: int = 0,
    limit: int = 100
):
    products = fake_products_db

    if category:
        products = [p for p in products if p.category == category]
    if min_price is not None:
        products = [p for p in products if p.price >= min_price]
    if max_price is not None:
        products = [p for p in products if p.price <= max_price]

    return products[skip: skip+limit]

@router.get('/{product_id}', response_model=Product)
def read_product(product_id: int):
    for product in fake_products_db:
        if product.id == product_id:
            return product
    raise HTTPException(status_code=404, detail="product not found")
