
STRUCTURED_SYSTEM_PROMPT = """
You are a helpful AI assistant.

Follow the user's instructions carefully.
Return information in the exact format requested.
Do not add unnecessary explanations outside the requested format.
"""


ARTICLE_SUMMARY_USER_PROMPT = """
Analyze the following article and create a structured summary.

Article:
{article}

{format_instructions}
"""


JSON_USER_PROMPT = """
Answer the following question.

Return your answer as valid JSON containing:
- answer: A short answer.
- explanation: A brief explanation.

Question:
{question}

Return only valid JSON.
"""


LIST_USER_PROMPT = """
Identify the programming languages mentioned in the following text.

Text:
{text}

Return the languages as a comma-separated list.
Do not include explanations.
"""
