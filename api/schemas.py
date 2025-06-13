from pydantic import BaseModel
from typing import List

class PricePoint(BaseModel):
    datetime: str
    price: float

class SignalResponse(BaseModel):
    datetime: List[str]
    trend: List[float]
    vol: List[float]
    signal: List[int]