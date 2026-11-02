from ai_proj.core.prompt_manager import PromptManager


def main():
    prompt_manager = PromptManager()

    messages = prompt_manager.build_messages(
        prompt="Explain Python decorators.",
        context="The user is a Python beginner.",
    )

    for message in messages:
        print(f"\nROLE: {message['role']}")
        print("-" * 50)
        print(message["content"])


if __name__ == "__main__":
    main()