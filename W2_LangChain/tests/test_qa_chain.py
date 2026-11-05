
from W2_LangChain.chains.qa_chain import create_qa_chain


def test_qa_chain():

    # Create the chain
    chain = create_qa_chain()

    # Provide input
    response = chain.invoke(
        {
            "question": "What is LangChain? Explain in simple words."
        }
    )

    print("\nQuestion:")
    print("What is LangChain?")

    print("\nAnswer:")
    print(response)


if __name__ == "__main__":
    test_qa_chain()
