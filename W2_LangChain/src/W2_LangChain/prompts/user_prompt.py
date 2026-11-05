
from langchain_core.prompts import PromptTemplate


USER_PROMPT = PromptTemplate.from_template(
    """
Answer the following question:

{question}
"""
)
