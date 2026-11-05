
from W2_LangChain.models.model_factory import get_chat_model


def test_chat_model():

    model = get_chat_model()

    response = model.invoke(
        "Explain LangChain in two simple sentences."
    )

    print("\nModel response:")
    print(response.content)


if __name__ == "__main__":
    test_chat_model()
