
from W2_LangChain.output_parsers.parsers import (
    get_string_parser,
    get_json_parser,
    get_pydantic_parser,
    get_list_parser,
)


def test_string_parser():
    parser = get_string_parser()

    result = parser.parse(
        "LangChain helps developers build LLM applications."
    )

    print("\nString Parser:")
    print(result)

    assert isinstance(result, str)


def test_json_parser():
    parser = get_json_parser()

    result = parser.parse(
        """
        {
            "answer": "LangChain",
            "explanation": "A framework for building LLM applications."
        }
        """
    )

    print("\nJSON Parser:")
    print(result)

    assert isinstance(result, dict)


def test_pydantic_parser():
    parser = get_pydantic_parser()

    result = parser.parse(
        """
        {
            "title": "Introduction to Python",
            "summary": "Python is a programming language.",
            "key_points": [
                "Easy to learn",
                "Widely used",
                "Supports multiple programming paradigms"
            ]
        }
        """
    )

    print("\nPydantic Parser:")
    print(result)

    print("\nTitle:", result.title)
    print("Summary:", result.summary)
    print("Key Points:", result.key_points)

    assert result.title == "Introduction to Python"


def test_list_parser():
    parser = get_list_parser()

    result = parser.parse(
        "Python, Java, JavaScript, C++"
    )

    print("\nList Parser:")
    print(result)

    assert isinstance(result, list)


if __name__ == "__main__":
    test_string_parser()
    test_json_parser()
    test_pydantic_parser()
    test_list_parser()

    print("\nAll parser tests passed!")
