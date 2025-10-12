"""Pydantic schema for product review analysis."""
from pydantic import BaseModel, Field, validator
from typing import Literal, List, Optional


class ReviewAnalysis(BaseModel):
    """Structured analysis of a product review."""
    
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence in sentiment classification"
    )
    
    key_themes: List[str] = Field(
        min_items=1,
        max_items=5,
        description="Main themes mentioned in the review"
    )
    
    product_issues: Optional[List[str]] = Field(
        default=None,
        description="Specific problems mentioned"
    )
    
    recommendations: List[str] = Field(
        min_items=1,
        description="Actionable recommendations based on the review"
    )
    
    @validator('key_themes', 'recommendations', 'product_issues')
    def no_empty_strings(cls, v):
        """Ensure list items are not empty or whitespace."""
        if v is not None and any(not item.strip() for item in v):
            raise ValueError(
                'List items cannot be empty or whitespace'
            )
        return v
    
    @validator('key_themes', 'recommendations')
    def no_duplicates(cls, v):
        """Ensure list items are unique."""
        if len(v) != len(set(v)):
            raise ValueError('List items must be unique')
        return v
