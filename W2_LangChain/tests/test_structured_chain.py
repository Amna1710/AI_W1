
from W2_LangChain.chains.structured_chain import (
    create_article_summary_chain,
)


def test_article_summary_chain():

    chain = create_article_summary_chain()

    article = """
    Python is a popular programming language.
    It was created by Guido van Rossum and first released in 1991.

    Python is widely used in web development, automation,
    data science, machine learning, and artificial intelligence.

    Its simple syntax makes it beginner-friendly.
    It also provides a large ecosystem of libraries.
    """

    result = chain.invoke(
        {
            "article": article
        }
    )

    print("\nStructured Article Summary:")
    print(result)

    print("\nTitle:", result.title)
    print("Summary:", result.summary)
    print("Key Points:")

    for point in result.key_points:
        print("-", point)


if __name__ == "__main__":
    test_article_summary_chain()
