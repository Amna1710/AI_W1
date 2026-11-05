# Document Q&A CLI

A Python command-line application for asking questions about PDF, TXT,
and DOCX documents. It uses Ollama to generate embeddings and answers,
NumPy for in-memory cosine-similarity search, and a Retrieval-Augmented
Generation (RAG) workflow.

## Features

-   Load PDF, TXT, and DOCX files.
-   Extract text and split it into overlapping chunks.
-   Generate embeddings with Ollama (`embeddinggemma` by default).
-   Retrieve relevant chunks using cosine similarity.
-   Generate answers with an Ollama chat model (`qwen3.5:4b` in the
    development setup).
-   Keep limited conversation history during the current session.
-   Add, list, and remove documents through the CLI.
-   Display source file names and chunk numbers with answers.
-   Support general conversation when no documents are loaded, provided
    `generate_general_answer()` is implemented and wired into the CLI.

## Project structure

``` text
W1_Proj/
├── documents/
├── src/document_qa/
│   ├── cli/interface.py
│   ├── core/config.py
│   ├── document/loader.py
│   ├── document/chunker.py
│   ├── embeddings/base.py
│   ├── embeddings/ollama_provider.py
│   ├── embeddings/embedding_service.py
│   ├── vector_store/in_memory.py
│   ├── llm/base.py
│   ├── llm/ollama_provider.py
│   ├── prompts/system_prompt.py
│   ├── prompts/few_shot_examples.py
│   ├── prompts/user_prompt.py
│   ├── services/answer_service.py
│   ├── services/conversation_memory.py
│   ├── services/document_manager.py
│   └── main.py
├── tests/
├── .env
├── .env.example
├── requirements.txt
├── README.md
└── DOCUMENTATION.md
```


## Requirements

-   Python (use one consistent interpreter for installation and
    execution).
-   Ollama installed and running.
-   Ollama models: `embeddinggemma` and `qwen3.5:4b` (or the model names
    configured in `.env`).

Install/pull the Ollama models:

``` powershell
ollama pull embeddinggemma
ollama pull qwen3.5:4b
ollama list
```

## Installation

From the project root in PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have not created `requirements.txt`, the baseline packages used
by this project are:

``` powershell
pip install httpx python-dotenv pymupdf python-docx numpy colorama
```

## Configuration

Create `.env` in the project root. Example:

``` env
OLLAMA_BASE_URL=http://127.0.0.1:11434
EMBEDDING_MODEL=embeddinggemma
LLM_MODEL=qwen3.5:4b
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SIMILARITY=0.50
CONVERSATION_MAX_TURNS=5
```

Ensure the variable names match `src/document_qa/core/config.py`. Do not
commit `.env` if it contains private values.

-   `OLLAMA_BASE_URL`: local Ollama API address.
-   `EMBEDDING_MODEL`: embedding model.
-   `LLM_MODEL`: chat model.
-   `RETRIEVAL_TOP_K`: maximum number of retrieved chunks.
-   `RETRIEVAL_MIN_SIMILARITY`: minimum accepted cosine similarity.
-   `CONVERSATION_MAX_TURNS`: retained conversation turns.

## Run

From the project root:

``` powershell
$env:PYTHONPATH = "src"
python -m document_qa.main
```

Use the CLI menus to add a document, ask questions, manage loaded
documents, or exit.

## How it works

1.  `DocumentLoader` extracts text from the selected file.
2.  `TextChunker` splits the text into smaller overlapping chunks.
3.  `EmbeddingService` sends chunks to Ollama to create vectors.
4.  `InMemoryVectorStore` stores chunks and vectors.
5.  A question is embedded using the same embedding model.
6.  The vector store ranks chunks by cosine similarity and returns the
    best matches.
7.  `AnswerService` builds a prompt containing the question and
    retrieved context.
8.  Ollama generates an answer, and the CLI displays it with source
    references.

## Troubleshooting

### No relevant chunks found

Temporarily use these diagnostic settings:

``` env
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SIMILARITY=0.0
```

Restart the application and inspect similarity scores. A zero threshold
is useful for diagnosis; it is not necessarily a good permanent setting.
Similarity scores are not probabilities.

### Ollama request times out

Check that Ollama is running and that the model is installed. Increase
the HTTP read timeout in `llm/ollama_provider.py`, for example:

``` python
timeout=httpx.Timeout(
    connect=10.0,
    read=300.0,
    write=30.0,
    pool=10.0,
)
```

A longer timeout gives the model more time; it does not make generation
faster.

### Scanned PDF

The baseline PDF loader extracts selectable text. Image-only/scanned
PDFs need OCR, which is not included in the baseline implementation.

## Limitations

-   Vector data is stored in memory and is lost when the program exits.
-   Conversation history is session-only.
-   Retrieval quality depends on document extraction, chunking,
    embeddings, and threshold settings.
-   OCR for scanned PDFs is not included.
-   Local model speed depends on the computer's available resources.

See [DOCUMENTATION.md]for architecture, module
responsibilities, workflows, testing guidance, and further details.
