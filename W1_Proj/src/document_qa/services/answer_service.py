
from document_qa.llm.base import LLMProvider

from document_qa.prompts.system_prompt import SYSTEM_PROMPT
from document_qa.prompts.few_shot_examples import get_few_shot_examples
from document_qa.prompts.user_prompt import build_user_prompt

from document_qa.vector_store.in_memory import SearchResult
from document_qa.services.conversation_memory import ConversationMemory


class AnswerService:
    """Generates document-grounded answers using conversation memory."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        conversation_memory: ConversationMemory,
    ) -> None:
        self.llm_provider = llm_provider
        self.conversation_memory = conversation_memory

    def _build_messages(
        self,
        question: str,
        search_results: list[SearchResult],
    ) -> list[dict[str, str]]:
        """Build messages for the LLM."""

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        # Add few-shot examples.
        for example in get_few_shot_examples():
            example_prompt = f"""
Retrieved Document Context:
{example["context"]}

Question:
{example["question"]}
"""

            messages.append({
                "role": "user",
                "content": example_prompt.strip(),
            })

            messages.append({
                "role": "assistant",
                "content": example["answer"],
            })

        # Add previous conversation history.
        messages.extend(self.conversation_memory.get_history())

        # Add current question and retrieved document context.
        actual_user_prompt = build_user_prompt(
            question=question,
            search_results=search_results,
        )

        messages.append({
            "role": "user",
            "content": actual_user_prompt,
        })

        return messages

    async def generate_answer(
        self,
        question: str,
        search_results: list[SearchResult],
    ) -> str:
        """Generate an answer using retrieved document context."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(search_results, list):
            raise ValueError("Search results must be a list.")

        if not search_results:
            return (
                "I could not find this information "
                "in the provided document."
            )

        messages = self._build_messages(
            question=question.strip(),
            search_results=search_results,
        )

        answer = await self.llm_provider.generate(messages)

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The LLM returned an empty answer.")

        answer = answer.strip()

        # Save only successfully generated answers.
        self.conversation_memory.add_turn(
            question=question.strip(),
            answer=answer,
        )

        return answer


    async def generate_general_answer(self, question: str) -> str:
        """Generate a general conversational answer without documents."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        messages = [
            {
                "role": "system",
                "content": """
You are a helpful, friendly, and knowledgeable AI assistant.

You can answer general questions, explain programming concepts,
discuss AI models, and respond naturally to greetings.

Rules:
1. Be conversational, clear, and helpful.
2. Answer using your general knowledge.
3. Do not claim to have access to uploaded documents.
4. If the user asks about a document, explain that they need
   to add the document first.
5. Never invent details about your own model or capabilities.
6. Keep answers concise unless more explanation is requested.
""",
            },
            {
                "role": "user",
                "content": question.strip(),
            },
        ]

        answer = await self.llm_provider.generate(messages)

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The LLM returned an empty answer.")

        answer = answer.strip()

        self.conversation_memory.add_turn(
            question=question.strip(),
            answer=answer,
        )

        return answer

