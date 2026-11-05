
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from W2_LangChain.models.model_factory import get_chat_model
from W2_LangChain.prompts.system_prompt import SYSTEM_PROMPT
from W2_LangChain.chains.components import (
    count_words,
    count_characters,
)


def create_reusable_analysis_chain():
    """
    Generate an answer and calculate its statistics.

    Flow:
        Input
          -> Prompt
          -> Chat Model
          -> String Parser
          -> Parallel Processing
    """

    # Step 1: Get the selected chat model
    model = get_chat_model()

    # Step 2: Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Explain the following topic in simple words:\n\n"
                "{topic}",
            ),
        ]
    )

    # Step 3: Create the answer-generation chain
    answer_chain = prompt | model | StrOutputParser()

    # Step 4: Create reusable Runnable components
    word_counter = RunnableLambda(count_words)
    character_counter = RunnableLambda(count_characters)

    # Step 5: Process the generated answer in parallel
    result_chain = answer_chain | RunnableParallel(
        answer=RunnablePassthrough(),
        word_count=word_counter,
        character_count=character_counter,
    )

    return result_chain
