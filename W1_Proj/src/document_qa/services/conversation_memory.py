
class ConversationMemory:
    """Stores conversation history in memory."""

    def __init__(self, max_turns: int = 5) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero.")

        self.max_turns = max_turns
        self.messages: list[dict[str, str]] = []

    def add_turn(self, question: str, answer: str) -> None:
        """Save one user question and its AI response."""

        self.messages.append({
            "role": "user",
            "content": question,
        })

        self.messages.append({
            "role": "assistant",
            "content": answer,
        })

        self.messages = self.messages[-(self.max_turns * 2):]

    def get_history(self) -> list[dict[str, str]]:
        """Return a copy of conversation history."""

        return self.messages.copy()

    def build_retrieval_query(
        self,
        question: str,
        recent_turns: int = 2,
    ) -> str:
        """Combine recent conversation with the current question."""

        if recent_turns < 0:
            raise ValueError("recent_turns cannot be negative.")

        if recent_turns == 0:
            return question.strip()

        recent_messages = self.messages[-(recent_turns * 2):]

        context_parts = []

        for message in recent_messages:
            role = message["role"].capitalize()
            content = message["content"]
            context_parts.append(f"{role}: {content}")

        conversation_context = "\n".join(context_parts)

        if not conversation_context:
            return question.strip()

        return (
            f"Previous conversation:\n"
            f"{conversation_context}\n\n"
            f"Current question:\n{question.strip()}"
        )

    def get_turn_count(self) -> int:
        """Return the number of stored conversation turns."""

        return len(self.messages) // 2

    def clear(self) -> None:
        """Clear conversation history."""

        self.messages.clear()
