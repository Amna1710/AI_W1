
import json
from pathlib import Path
from pprint import pprint

from W2_LangChain.chains.qa_chain import create_qa_chain
from W2_LangChain.chains.structured_chain import (
    create_article_summary_chain,
    create_json_chain,
    create_list_chain,
)
from W2_LangChain.pipelines import process_file


def print_separator():
    print("\n" + "=" * 60)



def ask_question():
    """Run the normal question-answering chain with streaming."""

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nGenerating answer...\n")
    print("AI Response:\n")

    chain = create_qa_chain()

    try:
        for chunk in chain.stream({"question": question}):
            print(chunk, end="", flush=True)

        print("\n")

    except Exception as error:
        print(f"\nError generating response: {error}")



def generate_article_summary():
    """Generate a structured article summary."""

    article = input("\nPaste your article text: ").strip()

    if not article:
        print("Please provide article text.")
        return

    print("\nGenerating article summary...")

    chain = create_article_summary_chain()
    response = chain.invoke({"article": article})

    print("\nArticle Summary:")
    pprint(response)


def generate_json():
    """Generate a JSON response."""

    question = input("\nEnter your request: ").strip()

    if not question:
        print("Please enter a request.")
        return

    print("\nGenerating JSON...")

    chain = create_json_chain()
    response = chain.invoke({"question": question})

    print("\nJSON Response:")
    print(json.dumps(response, indent=4, ensure_ascii=False))


def generate_list():
    """Generate a comma-separated list."""

    text = input("\nEnter a topic: ").strip()

    if not text:
        print("Please enter a topic.")
        return

    print("\nGenerating list...")

    chain = create_list_chain()
    response = chain.invoke({"text": text})

    print("\nGenerated List:")

    for index, item in enumerate(response, start=1):
        print(f"{index}. {item}")


def process_document():
    """Load, enrich, and split a document."""

    file_path = input("\nEnter document path: ").strip().strip('"')

    if not file_path:
        print("Please provide a document path.")
        return

    path = Path(file_path).expanduser()

    if not path.is_file():
        print(f"File not found: {path}")
        return

    splitter_type = input(
        "\nSelect splitter (recursive/character/token) [recursive]: "
    ).strip().lower()

    if not splitter_type:
        splitter_type = "recursive"

    try:
        chunk_size = int(
            input("Enter chunk size [1000]: ").strip() or "1000"
        )

        chunk_overlap = int(
            input("Enter chunk overlap [150]: ").strip() or "150"
        )

        print("\nProcessing document...")

        chunks = process_file(
            file_path=path,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        print(f"\nDocument processed successfully!")
        print(f"Total chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            print_separator()
            print(f"CHUNK {index}")
            print_separator()

            print("\nContent:")
            print(chunk.page_content)

            print("\nMetadata:")
            pprint(chunk.metadata)

    except ValueError as error:
        print(f"Invalid input: {error}")


def display_menu():
    """Display available project features."""

    print_separator()
    print("          W2 LANGCHAIN PROJECT")
    print_separator()

    print("1. Ask a Question (QA Chain)")
    print("2. Generate Article Summary")
    print("3. Generate JSON Response")
    print("4. Generate a List")
    print("5. Process a Document")
    print("0. Exit")

    print_separator()


def main():
    """Main application entry point."""

    print("\nWelcome to W2 LangChain!")

    while True:
        display_menu()

        choice = input("Select an option: ").strip()

        if choice == "0":
            print("\nThank you for using W2 LangChain!")
            break

        actions = {
            "1": ask_question,
            "2": generate_article_summary,
            "3": generate_json,
            "4": generate_list,
            "5": process_document,
        }

        action = actions.get(choice)

        if action is None:
            print("\nInvalid option. Please select a valid option.")
            continue

        try:
            action()

        except Exception as error:
            print(f"\nAn error occurred: {error}")

        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()
