
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from ai_proj.api.output_schema import (
    AIOutput,
    ExtractionOutput,
    ClassificationOutput,
    GenerationOutput,
)


UseCase = Literal[
    "question_answering",
    "extraction",
    "classification",
    "generation",
]


# Map each use case to its corresponding Pydantic output schema.
OUTPUT_SCHEMAS = {
    "question_answering": AIOutput,
    "extraction": ExtractionOutput,
    "classification": ClassificationOutput,
    "generation": GenerationOutput,
}


class StructuredOutputParser:

    @staticmethod
    def parse(
        response: str,
        schema: Any,
    ) -> Any:
        """
        Parse and validate an AI response using the given schema.
        """
        try:
            adapter = TypeAdapter(schema)

            return adapter.validate_json(response)

        except ValidationError as error:
            raise ValueError(
                f"Output validation failed: {error}"
            ) from error


def parse_ai_output(
    response: str,
    use_case: UseCase,
) -> Any:
    """
    Select the appropriate schema and validate the AI response.
    """
    schema = OUTPUT_SCHEMAS.get(use_case)

    if schema is None:
        raise ValueError(
            f"No output schema configured for: {use_case}"
        )

    return StructuredOutputParser.parse(
        response=response,
        schema=schema,
    )
