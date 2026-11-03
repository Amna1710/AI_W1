
import asyncio
import logging

import httpx

from document_qa.cli.interface import CLIInterface
from document_qa.core.config import settings
from document_qa.document.chunker import TextChunker
from document_qa.embeddings.ollama_provider import OllamaEmbeddingProvider
from document_qa.embeddings.embedding_service import EmbeddingService
from document_qa.vector_store.in_memory import InMemoryVectorStore
from document_qa.llm.ollama_provider import OllamaLLMProvider
from document_qa.services.answer_service import AnswerService
from document_qa.services.conversation_memory import ConversationMemory
from document_qa.services.document_manager import DocumentManager


logger = logging.getLogger(__name__)


# ==================================================
# ASK QUESTIONS
# ==================================================


async def ask_question(
    cli: CLIInterface,
    document_manager: DocumentManager,
    embedding_service: EmbeddingService,
    vector_store: InMemoryVectorStore,
    answer_service: AnswerService,
    conversation_memory: ConversationMemory,
) -> None:
    """Handle general conversation and document question-answering."""

    while True:

        question = cli.get_question()

        if not question.strip():
            cli.display_error("Please enter a question.")
            continue

        if question.strip().lower() in {"back", "menu", "exit"}:
            cli.display_message("Returning to main menu...")
            return

        try:

            # ------------------------------------------
            # GENERAL CONVERSATION (NO DOCUMENTS)
            # ------------------------------------------

            if document_manager.get_document_count() == 0:

                cli.display_message(
                    "Generating response with Ollama. Please wait..."
                )

                answer = await answer_service.generate_general_answer(
                    question
                )

                cli.display_answer(answer)

                continue

            # ------------------------------------------
            # DOCUMENT QUESTION ANSWERING
            # ------------------------------------------

            cli.display_message(
                "Generating question embedding..."
            )

            # Use the actual question for retrieval.
            retrieval_query = question

            query_embedding = await embedding_service.embed_query(
                retrieval_query
            )

            # ------------------------------------------
            # SEARCH VECTOR STORE
            # ------------------------------------------

            results = vector_store.search(
                query_embedding=query_embedding,
                top_k=settings.RETRIEVAL_TOP_K,
                min_similarity=settings.RETRIEVAL_MIN_SIMILARITY,
            )

            cli.display_search_results(results)

            # ------------------------------------------
            # HANDLE NO SEARCH RESULTS
            # ------------------------------------------

            if not results:

                answer = (
                    "I could not find sufficiently relevant "
                    "information in the loaded documents."
                )

                cli.display_answer(answer)
                cli.display_sources(results)

                continue

            # ------------------------------------------
            # GENERATE DOCUMENT-BASED ANSWER
            # ------------------------------------------

            cli.display_message(
                "Generating answer with Ollama. Please wait..."
            )

            answer = await answer_service.generate_answer(
                question=question,
                search_results=results,
            )

            # ------------------------------------------
            # DISPLAY ANSWER AND SOURCES
            # ------------------------------------------

            cli.display_answer(answer)
            cli.display_sources(results)

        except (
            ValueError,
            httpx.HTTPError,
        ) as exc:

            logger.exception("Question answering failed.")

            cli.display_error(
                f"Error generating answer: {exc}"
            )


# ==================================================
# DOCUMENT MANAGEMENT
# ==================================================

async def manage_documents(
    cli: CLIInterface,
    document_manager: DocumentManager,
    conversation_memory: ConversationMemory,
) -> str | None:
    """Handle document management operations."""

    while True:

        action = cli.get_document_action()

        # ------------------------------------------
        # ADD DOCUMENT
        # ------------------------------------------

        if action == "1":

            document_path = cli.get_document_path()

            if not document_path:
                cli.display_error("Please enter a document path.")
                continue

            try:

                cli.display_message(
                    "Loading and processing document. Please wait..."
                )

                managed_document = await document_manager.add_document(
                    document_path
                )

                # Clear conversation context because
                # the document collection has changed.
                conversation_memory.clear()

                cli.display_document_info(
                    managed_document.document
                )

                cli.display_chunk_info(
                    managed_document.chunks
                )

                cli.display_embedding_info(
                    managed_document.chunks
                )

                cli.display_success(
                    f"Successfully added "
                    f"{managed_document.document.file_name}."
                )

                cli.display_document_management_summary(
                    document_manager.get_document_count(),
                    document_manager.get_chunk_count(),
                )

                # Ask the user what to do next.
                next_action = cli.get_next_action()

                if next_action == "1":
                    return "ask"

                elif next_action == "2":
                    continue

                elif next_action == "3":
                    return "menu"

                else:
                    cli.display_error(
                        "Invalid option. Please select 1, 2, or 3."
                    )
                    continue

            except (
                FileNotFoundError,
                ValueError,
                OSError,
                httpx.HTTPError,
            ) as exc:

                logger.exception("Document loading failed.")

                cli.display_error(
                    f"Error adding document: {exc}"
                )

        # ------------------------------------------
        # LIST DOCUMENTS
        # ------------------------------------------

        elif action == "2":

            documents = document_manager.list_documents()

            cli.display_documents(documents)

            cli.display_document_management_summary(
                document_manager.get_document_count(),
                document_manager.get_chunk_count(),
            )

        # ------------------------------------------
        # REMOVE DOCUMENT
        # ------------------------------------------

        elif action == "3":

            documents = document_manager.list_documents()

            document_path = cli.get_document_to_remove(
                documents
            )

            if not document_path:
                continue

            if cli.confirm_action(
                "Are you sure you want to remove this document?"
            ):

                removed = document_manager.remove_document(
                    document_path
                )

                if removed:

                    conversation_memory.clear()

                    cli.display_success(
                        "Document removed successfully."
                    )

                    cli.display_document_management_summary(
                        document_manager.get_document_count(),
                        document_manager.get_chunk_count(),
                    )

                else:

                    cli.display_error(
                        "Document not found. Please check the full path."
                    )

        # ------------------------------------------
        # CLEAR ALL DOCUMENTS
        # ------------------------------------------

        elif action == "4":

            if cli.confirm_action(
                "Are you sure you want to clear all documents?"
            ):

                document_manager.clear_documents()
                conversation_memory.clear()

                cli.display_success(
                    "All documents and their embeddings have been cleared."
                )

                cli.display_document_management_summary(
                    document_manager.get_document_count(),
                    document_manager.get_chunk_count(),
                )

        # ------------------------------------------
        # BACK TO MAIN MENU
        # ------------------------------------------

        elif action == "5":
            return "menu"

        else:

            cli.display_error(
                "Invalid option. Please select 1, 2, 3, 4, or 5."
            )


# ==================================================
# MAIN APPLICATION
# ==================================================

async def main() -> None:

    cli = CLIInterface()

    # ----------------------------------------------
    # INITIALIZE DOCUMENT CHUNKER
    # ----------------------------------------------

    chunker = TextChunker(
        chunk_size=1000,
        overlap=200,
    )

    # ----------------------------------------------
    # INITIALIZE OLLAMA CONNECTION
    # ----------------------------------------------

    base_url = settings.OLLAMA_BASE_URL

    # ----------------------------------------------
    # INITIALIZE EMBEDDING PROVIDER
    # ----------------------------------------------

    embedding_provider = OllamaEmbeddingProvider(
        model=settings.EMBEDDING_MODEL,
        base_url=base_url,
    )

    # ----------------------------------------------
    # INITIALIZE LLM PROVIDER
    # ----------------------------------------------

    llm_provider = OllamaLLMProvider(
        model=settings.LLM_MODEL,
        base_url=base_url,
    )

    # ----------------------------------------------
    # INITIALIZE EMBEDDING SERVICE
    # ----------------------------------------------

    embedding_service = EmbeddingService(
        provider=embedding_provider,
        batch_size=32,
    )

    # ----------------------------------------------
    # INITIALIZE IN-MEMORY VECTOR STORE
    # ----------------------------------------------

    vector_store = InMemoryVectorStore()

    # ----------------------------------------------
    # INITIALIZE CONVERSATION MEMORY
    # ----------------------------------------------

    conversation_memory = ConversationMemory(
        max_turns=settings.CONVERSATION_MAX_TURNS
    )

    # ----------------------------------------------
    # INITIALIZE ANSWER SERVICE
    # ----------------------------------------------

    answer_service = AnswerService(
        llm_provider=llm_provider,
        conversation_memory=conversation_memory,
    )

    # ----------------------------------------------
    # INITIALIZE DOCUMENT MANAGER
    # ----------------------------------------------

    document_manager = DocumentManager(
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    cli.display_welcome()

    try:

        while True:

            cli.display_menu()

            choice = cli.get_choice()

            # ======================================
            # ASK A QUESTION
            # ======================================

            if choice == "1":

                await ask_question(
                    cli=cli,
                    document_manager=document_manager,
                    embedding_service=embedding_service,
                    vector_store=vector_store,
                    answer_service=answer_service,
                    conversation_memory=conversation_memory,
                )

            # ======================================
            # DOCUMENT MANAGEMENT
            # ======================================

            elif choice == "2":

                action = await manage_documents(
                    cli=cli,
                    document_manager=document_manager,
                    conversation_memory=conversation_memory,
                )

                if action == "ask":

                    # Start question-answering directly.
                    # Do not return to the main menu first.

                    await ask_question(
                        cli=cli,
                        document_manager=document_manager,
                        embedding_service=embedding_service,
                        vector_store=vector_store,
                        answer_service=answer_service,
                        conversation_memory=conversation_memory,
                    )

            # ======================================
            # EXIT APPLICATION
            # ======================================

            elif choice == "3":

                cli.display_success(
                    "Thank you for using Document Q&A Assistant!"
                )

                break

            # ======================================
            # INVALID OPTION
            # ======================================

            else:

                cli.display_error(
                    "Invalid choice. Please select 1, 2, or 3."
                )

    finally:

        # Close both HTTP clients when the application exits.
        await embedding_provider.close()
        await llm_provider.close()


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    asyncio.run(main())
