from pydantic import BaseModel
from typing import List, Optional

class PricePoint(BaseModel):
    datetime: str
    price: float
    bid_depth: Optional[float] = None
    ask_depth: Optional[float] = None

class SignalResponse(BaseModel):
    datetime: List[str]
    prob_up: List[float]
    signal: List[int]
