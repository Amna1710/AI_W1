
from sqlalchemy import create_engine, text
from langchain_core.documents import Document


class DatabaseDocumentLoader:
    """Custom loader for database query results."""

    def __init__(self, connection_url: str):
        if not connection_url:
            raise ValueError(
                "Database connection URL cannot be empty."
            )

        self.engine = create_engine(connection_url)

    def load(
        self,
        query: str,
        params: dict | None = None,
        source_name: str = "database",
    ) -> list[Document]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("SQL query cannot be empty.")

        documents = []

        with self.engine.connect() as connection:

            result = connection.execute(
                text(query),
                params or {},
            )

            for row_index, row in enumerate(
                result.mappings(),
                start=1,
            ):

                row_data = dict(row)

                content = "\n".join(
                    f"{key}: {value}"
                    for key, value in row_data.items()
                )

                document = Document(
                    page_content=content,
                    metadata={
                        "source": source_name,
                        "file_type": "database",
                        "row_index": row_index,
                    },
                )

                documents.append(document)

        return documents

    def close(self):
        """Dispose of the database connection pool."""
        self.engine.dispose()
