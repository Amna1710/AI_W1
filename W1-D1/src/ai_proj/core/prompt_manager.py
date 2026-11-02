
import json
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined


UseCase = Literal[
    "question_answering",
    "extraction",
    "classification",
    "generation",
]


class PromptManager:

    def __init__(self) -> None:

        # Templates directory
        self.templates_dir = (
            Path(__file__).resolve().parent.parent / "templates"
        )

        # Initialize Jinja2
        self.environment = Environment(
            loader=FileSystemLoader(self.templates_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Load few-shot examples
        examples_path = (
            Path(__file__).resolve().parent / "few_shot.json"
        )

        with open(examples_path, "r", encoding="utf-8") as file:
            self.examples = json.load(file)

        # Validate examples structure
        if not isinstance(self.examples, dict):
            raise ValueError(
                "few_shot.json must contain a JSON object "
                "with use cases as keys."
            )

    def render(
        self,
        template_name: str,
        **context: Any,
    ) -> str:

        template = self.environment.get_template(template_name)

        return template.render(**context).strip()

    def build_messages(
        self,
        use_case: UseCase,
        user_input: str,
    ) -> list[dict[str, str]]:

        # Validate use case
        if use_case not in self.examples:
            raise ValueError(
                f"Unsupported use case: {use_case}. "
                f"Available use cases: {list(self.examples.keys())}"
            )

        # Build system prompt
        system_message = self.render(
            f"{use_case}.j2",
            examples=self.examples[use_case],
        )

        # Build conversation messages
        return [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]
