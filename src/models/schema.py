from typing import List, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class Column(BaseModel):
    """Represents a column in a data schema."""

    name: str
    type: str
    description: str


class DataSchema(BaseModel):
    """Represents a complete data schema."""

    table_name: str
    description: str
    columns: List[Column]
