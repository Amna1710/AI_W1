
from langchain_core.prompts import ChatPromptTemplate

from W2_LangChain.models.model_factory import get_chat_model

from W2_LangChain.output_parsers.parsers import (
    get_pydantic_parser,
    get_json_parser,
    get_list_parser,
)

from W2_LangChain.prompts.structured_prompts import (
    STRUCTURED_SYSTEM_PROMPT,
    ARTICLE_SUMMARY_USER_PROMPT,
    JSON_USER_PROMPT,
    LIST_USER_PROMPT,
)


def create_article_summary_chain():
    """Create a chain that returns a validated ArticleSummary."""

    model = get_chat_model()
    parser = get_pydantic_parser()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STRUCTURED_SYSTEM_PROMPT),
            ("human", ARTICLE_SUMMARY_USER_PROMPT),
        ]
    )

    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | model | parser

    return chain


def create_json_chain():
    """Create a chain that returns a Python dictionary."""

    model = get_chat_model()
    parser = get_json_parser()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STRUCTURED_SYSTEM_PROMPT),
            ("human", JSON_USER_PROMPT),
        ]
    )

    chain = prompt | model | parser

    return chain


def create_list_chain():
    """Create a chain that returns a Python list."""

    model = get_chat_model()
    parser = get_list_parser()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STRUCTURED_SYSTEM_PROMPT),
            (
                "human",
                LIST_USER_PROMPT + "\n{format_instructions}",
            ),
        ]
    )

    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | model | parser

    return chain
