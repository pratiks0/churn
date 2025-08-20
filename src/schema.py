from typing import List, Optional
from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract_type: str
    payment_method: str
    internet_service: str
    streaming: bool
    tech_support: bool
    support_calls: int = Field(..., ge=0)
    num_services: int = Field(..., ge=0, le=10)

class PredictRequest(BaseModel):
    records: List[CustomerFeatures]

class PredictResponseItem(BaseModel):
    churn_probability: float
    churn_label: int

class PredictResponse(BaseModel):
    predictions: List[PredictResponseItem]
