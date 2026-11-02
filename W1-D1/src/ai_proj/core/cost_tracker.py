
from decimal import Decimal, ROUND_HALF_UP


# Prices are USD per 1 million tokens.
# Update these values when provider pricing changes.
MODEL_PRICING = {
    "gpt-4.1-mini": {
        "input": Decimal("0.40"),
        "output": Decimal("1.60"),
    },
}

OLLAMA_PRICING = {
    "input": Decimal("0"),
    "output": Decimal("0"),
}


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:

    if provider == "ollama":
        pricing = OLLAMA_PRICING

    elif provider == "openai":
        pricing = MODEL_PRICING.get(model)

        if pricing is None:
            return {
                "input_cost": None,
                "output_cost": None,
                "total_cost": None,
                "currency": "USD",
                "pricing_available": False,
            }

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    input_cost = (
        Decimal(input_tokens) * pricing["input"] / Decimal("1000000")
    )

    output_cost = (
        Decimal(output_tokens) * pricing["output"] / Decimal("1000000")
    )

    total_cost = input_cost + output_cost

    def money(value):
        return float(
            value.quantize(
                Decimal("0.00000001"),
                rounding=ROUND_HALF_UP,
            )
        )

    return {
        "input_cost": money(input_cost),
        "output_cost": money(output_cost),
        "total_cost": money(total_cost),
        "currency": "USD",
        "pricing_available": True,
    }
