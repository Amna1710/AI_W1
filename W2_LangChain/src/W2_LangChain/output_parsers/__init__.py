
from pydantic import BaseModel, Field

from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    PydanticOutputParser,
)


# Define the expected structure of an article summary
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")

    summary: str = Field(
        description="A short summary of the article"
    )

    key_points: list[str] = Field(
        description="Important points from the article"
    )


# Plain text parser
def get_string_parser():
    return StrOutputParser()


# JSON parser
def get_json_parser():
    return JsonOutputParser()


# Pydantic parser
def get_pydantic_parser():
    return PydanticOutputParser(
        pydantic_object=ArticleSummary
    )


# Comma-separated list parser
def get_list_parser():
    return CommaSeparatedListOutputParser()
