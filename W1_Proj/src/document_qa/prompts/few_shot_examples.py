
def get_few_shot_examples() -> list[dict[str, str]]:
    """
    Return generic example conversations demonstrating
    the expected document question-answering behavior.
    """

    return [
        {
            "context": """
Source: sample_document.pdf
Chunk: 1

The company was established in 2015.
Its headquarters are located in London.
""",
            "question": "When was the company established?",
            "answer": (
                "The company was established in 2015. "
                "[Source: sample_document.pdf, Chunk: 1]"
            ),
        },
        {
            "context": """
Source: sample_document.pdf
Chunk: 2

Available programming languages:
- Python
- Java
- JavaScript
""",
            "question": "Which programming languages are mentioned?",
            "answer": (
                "The document mentions three programming languages:\n"
                "- Python\n"
                "- Java\n"
                "- JavaScript\n"
                "[Source: sample_document.pdf, Chunk: 2]"
            ),
        },
        {
            "context": """
Source: sample_document.pdf
Chunk: 3

The application supports PDF document uploads.

Source: sample_document.pdf
Chunk: 4

The application also supports TXT and DOCX uploads.
""",
            "question": "Which document formats are supported?",
            "answer": (
                "The application supports three document formats: "
                "PDF, TXT, and DOCX.\n"
                "[Source: sample_document.pdf, Chunk: 3] "
                "[Source: sample_document.pdf, Chunk: 4]"
            ),
        },
        {
            "context": """
Source: sample_document.pdf
Chunk: 5

The company provides software development services.
""",
            "question": "What is the company's annual revenue?",
            "answer": (
                "I could not find this information in the provided document."
            ),
        },
    ]
