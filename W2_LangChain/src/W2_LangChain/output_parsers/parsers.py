
from pydantic import BaseModel, Field

from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    PydanticOutputParser,
)


class ArticleSummary(BaseModel):
    title: str = Field(
        description="Title of the article"
    )

    summary: str = Field(
        description="A short summary of the article"
    )

    key_points: list[str] = Field(
        description="Important points from the article"
    )


def get_string_parser():
    """Return a parser for plain text."""
    return StrOutputParser()


def get_json_parser():
    """Return a parser for JSON responses."""
    return JsonOutputParser()


def get_pydantic_parser():
    """Return a parser for ArticleSummary objects."""
    return PydanticOutputParser(
        pydantic_object=ArticleSummary
    )


def get_list_parser():
    """Return a parser for comma-separated lists."""
    return CommaSeparatedListOutputParser()
