
from W2_LangChain.chains.components import (
    count_words,
    count_characters,
    clean_text,
    get_text_statistics,
)

from W2_LangChain.chains.reusable_chain import (
    create_reusable_analysis_chain,
)


def test_python_components():

    text = "LangChain makes building AI applications easier."

    print("\n--- Testing Python Components ---")

    print("Word count:", count_words(text))
    print("Character count:", count_characters(text))

    print("Clean text:", clean_text("  Hello    LangChain  "))

    print("Statistics:", get_text_statistics(text))

    assert count_words(text) == 6
    assert count_characters(text) == len(text)
    assert clean_text("  Hello    LangChain  ") == "Hello LangChain"


def test_reusable_chain():

    print("\n--- Testing Reusable LCEL Chain ---")

    chain = create_reusable_analysis_chain()

    result = chain.invoke(
        {
            "topic": "What is LangChain and why is it useful?"
        }
    )

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nWord Count:")
    print(result["word_count"])

    print("\nCharacter Count:")
    print(result["character_count"])

    assert isinstance(result["answer"], str)
    assert isinstance(result["word_count"], int)
    assert isinstance(result["character_count"], int)


if __name__ == "__main__":

    test_python_components()
    test_reusable_chain()

    print("\nAll reusable component tests passed!")
