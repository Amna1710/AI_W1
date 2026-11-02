from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AIOutput(BaseModel):
    """General AI response."""

    model_config = ConfigDict(extra="forbid")

    answer: str


class UserOutput(BaseModel):
    """User information."""

    model_config = ConfigDict(extra="forbid")

    name: str
    age: int = Field(gt=0)


class ProductOutput(BaseModel):
    """Product information."""

    model_config = ConfigDict(extra="forbid")

    name: str
    price: float
    in_stock: bool


# ==========================================
# EXTRACTION OUTPUT
# ==========================================

class ExtractionOutput(BaseModel):
    """Structured information extracted from text."""

    model_config = ConfigDict(extra="forbid")

    extracted_data: dict[str, Any]


# ==========================================
# CLASSIFICATION OUTPUT
# ==========================================

class ClassificationOutput(BaseModel):
    """Classification result."""

    model_config = ConfigDict(extra="forbid")

    category: str
    confidence: float = Field(ge=0, le=1)
    reason: str


# ==========================================
# GENERATION OUTPUT
# ==========================================

class GenerationOutput(BaseModel):
    """Generated AI content."""

    model_config = ConfigDict(extra="forbid")

    content: str