
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from W2_LangChain.models.model_factory import get_chat_model
from W2_LangChain.prompts.system_prompt import SYSTEM_PROMPT
from W2_LangChain.prompts.user_prompt import USER_PROMPT


def create_qa_chain():
    """
    Create a question-answering chain using LCEL.

    Flow:
        System Prompt + User Prompt
        -> Chat Model
        -> String Output Parser
    """

    # Step 1: Create the chat model
    model = get_chat_model()

    # Step 2: Create the chat prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT.template),
        ]
    )

    # Step 3: Create the output parser
    output_parser = StrOutputParser()

    # Step 4: Connect components using LCEL
    chain = prompt | model | output_parser

    return chain
