
from colorama import Fore, Style, init


# Initialize terminal colors.
init(autoreset=True)


class CLIInterface:
    """Handles command-line user interaction."""

    # --------------------------------------------------
    # REUSABLE MENU DISPLAY
    # --------------------------------------------------

    @staticmethod
    def _display_menu_box(title: str, options: list) -> None:
        """Display a consistently aligned ASCII menu."""

        width = 44

        print()
        print("+" + "-" * (width + 2) + "+")

        title_text = f" {title.upper()} "
        title_padding = width - len(title_text)
        left_padding = title_padding // 2
        right_padding = title_padding - left_padding

        print(
            Fore.CYAN + Style.BRIGHT
            + "|"
            + " " * (left_padding + 1)
            + title_text
            + " " * (right_padding + 1)
            + "|"
        )

        print("+" + "-" * (width + 2) + "+")

        for label, color in options:

            plain_text = f"  {label}"
            padding = width - len(plain_text)

            print(
                Fore.WHITE
                + "| "
                + color
                + plain_text
                + " " * max(0, padding)
                + Fore.WHITE
                + " |"
            )

        print("|" + " " * (width + 2) + "|")
        print("+" + "-" * (width + 2) + "+")

    # --------------------------------------------------
    # GENERAL DISPLAY METHODS
    # --------------------------------------------------

    @staticmethod
    def display_welcome() -> None:

        print()
        print(Fore.CYAN + Style.BRIGHT + "+" + "-" * 54 + "+")
        print(Fore.CYAN + Style.BRIGHT + "|" + " " * 54 + "|")
        print(
            Fore.CYAN + Style.BRIGHT
            + "|"
            + " " * 9
            + "DOCUMENT Q&A ASSISTANT"
            + " " * 23
            + "|"
        )
        print(Fore.CYAN + Style.BRIGHT + "|" + " " * 54 + "|")
        print(
            Fore.WHITE
            + "|"
            + " " * 6
            + "Ask questions. Explore your documents."
            + " " * 10
            + "|"
        )
        print(Fore.CYAN + Style.BRIGHT + "|" + " " * 54 + "|")
        print(Fore.CYAN + Style.BRIGHT + "+" + "-" * 54 + "+")
        print()

    @staticmethod
    def display_section(title: str) -> None:

        print()
        print(
            Fore.CYAN + Style.BRIGHT
            + f"\n--- {title.upper()} ---"
        )
        print()

    @staticmethod
    def display_message(message: str) -> None:

        print(Fore.YELLOW + f"\n  [INFO] {message}")

    @staticmethod
    def display_success(message: str) -> None:

        print(
            Fore.GREEN + Style.BRIGHT
            + f"\n  [SUCCESS] {message}"
        )

    @staticmethod
    def display_error(message: str) -> None:

        print(
            Fore.RED + Style.BRIGHT
            + f"\n  [ERROR] {message}"
        )

    # --------------------------------------------------
    # MAIN MENU
    # --------------------------------------------------

    @staticmethod
    def display_menu() -> None:

        CLIInterface._display_menu_box(
            "MAIN MENU",
            [
                ("[1]  Ask a question", Fore.GREEN),
                ("[2]  Document management", Fore.GREEN),
                ("[3]  Exit", Fore.RED),
            ],
        )

    @staticmethod
    def get_choice() -> str:

        return input(
            Fore.YELLOW + Style.BRIGHT
            + "\n  Select an option: "
        ).strip()

    # --------------------------------------------------
    # DOCUMENT INPUT
    # --------------------------------------------------

    @staticmethod
    def get_document_path() -> str:

        return input(
            Fore.YELLOW + "\n  Enter document path: "
        ).strip().strip('"')

    @staticmethod
    def get_question() -> str:

        print(
            Fore.YELLOW
            + "  (Type 'back' to return to the main menu.)"
        )

        return input(
            Fore.GREEN + Style.BRIGHT + "\n  You: "
        ).strip()

    # --------------------------------------------------
    # ANSWER DISPLAY
    # --------------------------------------------------

    @staticmethod
    def display_answer(answer: str) -> None:

        print()
        print(
            Fore.CYAN + Style.BRIGHT
            + "+" + "-" * 48 + "+"
        )
        print(
            Fore.CYAN + Style.BRIGHT
            + "|"
            + " ASSISTANT RESPONSE ".center(48)
            + "|"
        )
        print(
            Fore.CYAN + Style.BRIGHT
            + "+" + "-" * 48 + "+"
        )

        print(Fore.WHITE + f"\n  {answer}\n")

        print(
            Fore.CYAN + Style.BRIGHT
            + "+" + "-" * 48 + "+"
        )

    # --------------------------------------------------
    # DOCUMENT INFORMATION
    # --------------------------------------------------

    @staticmethod
    def display_document_info(document) -> None:

        CLIInterface.display_section(
            "Document loaded successfully"
        )

        print(Fore.WHITE + f"  File name : {document.file_name}")
        print(Fore.WHITE + f"  File type : {document.file_type}")
        print(Fore.WHITE + f"  File path : {document.file_path}")

        if document.page_count is not None:
            print(Fore.WHITE + f"  Pages     : {document.page_count}")

        print(
            Fore.WHITE
            + f"  Characters: {len(document.content):,}"
        )

        CLIInterface.display_success(
            "Document loaded successfully."
        )

    @staticmethod
    def display_chunk_info(chunks) -> None:

        CLIInterface.display_section("Document chunking")

        print(Fore.WHITE + f"  Total chunks: {len(chunks)}")

        if chunks:

            print(
                Fore.CYAN + Style.BRIGHT
                + "\n  First chunk preview:"
            )

            print(Fore.WHITE + "  " + "-" * 48)
            print(Fore.WHITE + f"  {chunks[0].content[:300]}")
            print(Fore.WHITE + "  " + "-" * 48)

            print(
                Fore.CYAN + Style.BRIGHT
                + "\n  Chunk details:"
            )

            for chunk in chunks[:5]:

                print(
                    Fore.WHITE
                    + f"  Chunk {chunk.index}: "
                    + f"{len(chunk.content)} characters"
                )

            if len(chunks) > 5:

                print(
                    Fore.YELLOW
                    + f"  ... and {len(chunks) - 5} more chunks"
                )

    @staticmethod
    def display_embedding_info(chunks) -> None:

        embedded_count = sum(
            chunk.embedding is not None
            for chunk in chunks
        )

        CLIInterface.display_section("Embeddings generated")

        print(Fore.WHITE + f"  Total chunks     : {len(chunks)}")
        print(Fore.WHITE + f"  Embedded chunks  : {embedded_count}")

        if chunks and chunks[0].embedding is not None:

            print(
                Fore.WHITE
                + f"  Vector dimensions: {len(chunks[0].embedding)}"
            )

            print(
                Fore.CYAN + Style.BRIGHT
                + "\n  First vector preview:"
            )

            print(
                Fore.WHITE
                + f"  {chunks[0].embedding[:10]}"
            )

        CLIInterface.display_success(
            "Embedding generation completed."
        )

    # --------------------------------------------------
    # SEARCH RESULTS
    # --------------------------------------------------

    @staticmethod
    def display_search_results(results) -> None:

        CLIInterface.display_section("Search results")

        if not results:

            print(
                Fore.YELLOW
                + "  No relevant document chunks found."
            )
            return

        for rank, result in enumerate(results, start=1):

            print(
                Fore.CYAN + Style.BRIGHT
                + f"\n  RESULT {rank}"
            )

            print(
                Fore.GREEN
                + f"  Similarity score: {result.score:.4f}"
            )

            print(
                Fore.WHITE
                + f"  Source: {result.chunk.source_file}"
            )

            print(
                Fore.WHITE
                + f"  Chunk: {result.chunk.index}"
            )

            print(Fore.WHITE + "  " + "-" * 55)
            print(Fore.WHITE + f"  {result.chunk.content}")
            print(Fore.WHITE + "  " + "-" * 55)

    # --------------------------------------------------
    # DOCUMENT MANAGEMENT MENU
    # --------------------------------------------------

    @staticmethod
    def get_document_action() -> str:

        CLIInterface._display_menu_box(
            "DOCUMENT MANAGEMENT",
            [
                ("[1]  Add document", Fore.GREEN),
                ("[2]  List documents", Fore.GREEN),
                ("[3]  Remove document", Fore.GREEN),
                ("[4]  Clear all documents", Fore.GREEN),
                ("[5]  Back to main menu", Fore.RED),
            ],
        )

        return input(
            Fore.YELLOW + Style.BRIGHT
            + "\n  Select an option: "
        ).strip()

    @staticmethod
    def display_documents(documents) -> None:

        if not documents:

            print(
                Fore.YELLOW
                + "\n  No documents are currently loaded."
            )
            return

        CLIInterface.display_section("Loaded documents")

        for index, managed in enumerate(documents, start=1):

            document = managed.document

            print(
                Fore.GREEN + Style.BRIGHT
                + f"\n  [{index}] {document.file_name}"
            )

            print(
                Fore.WHITE
                + f"      Type   : {document.file_type}"
            )

            print(
                Fore.WHITE
                + f"      Chunks : {len(managed.chunks)}"
            )

            print(
                Fore.WHITE
                + f"      Path   : {document.file_path}"
            )

            if document.page_count is not None:

                print(
                    Fore.WHITE
                    + f"      Pages  : {document.page_count}"
                )

    @staticmethod
    def get_document_to_remove(documents) -> str:

        if not documents:

            print(
                Fore.YELLOW
                + "\n  No documents available to remove."
            )
            return ""

        CLIInterface.display_documents(documents)

        return input(
            Fore.YELLOW
            + "\n  Enter the full document path to remove: "
        ).strip().strip('"')

    # --------------------------------------------------
    # CONFIRMATION AND SUMMARY
    # --------------------------------------------------

    @staticmethod
    def confirm_action(message: str) -> bool:

        choice = input(
            Fore.YELLOW + f"\n  {message} (y/n): "
        ).strip().lower()

        return choice == "y"

    @staticmethod
    def display_document_management_summary(
        document_count: int,
        chunk_count: int,
    ) -> None:

        CLIInterface.display_section(
            "Document collection summary"
        )

        print(
            Fore.WHITE
            + f"  Loaded documents: {document_count}"
        )

        print(
            Fore.WHITE
            + f"  Total chunks    : {chunk_count}"
        )

    # --------------------------------------------------
    # SOURCE CITATIONS
    # --------------------------------------------------

    @staticmethod
    def display_sources(search_results) -> None:

        print(
            Fore.CYAN + Style.BRIGHT
            + "\n  RETRIEVED SOURCES"
        )

        if not search_results:

            print(
                Fore.YELLOW
                + "  No sources found."
            )
            return

        for index, result in enumerate(search_results, start=1):

            chunk = result.chunk

            print(
                Fore.GREEN
                + f"\n  [{index}] "
                + Fore.WHITE
                + chunk.source_file
            )

            print(
                Fore.WHITE
                + f"      Chunk: {chunk.index}"
            )

            print(
                Fore.WHITE
                + f"      Similarity: {result.score:.4f}"
            )

    # --------------------------------------------------
    # NEXT ACTION MENU
    # --------------------------------------------------

    @staticmethod
    def get_next_action() -> str:

        CLIInterface._display_menu_box(
            "NEXT ACTION",
            [
                ("[1]  Ask a question", Fore.GREEN),
                ("[2]  Continue document management", Fore.GREEN),
                ("[3]  Main menu", Fore.RED),
            ],
        )

        return input(
            Fore.YELLOW + Style.BRIGHT
            + "\n  Select an option: "
        ).strip()
