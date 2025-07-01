import tiktoken

price_per_token = {
    # GPT-4 family
    "gpt-4": 0.01,
    "gpt-4-0314": 0.01,
    "gpt-4-0613": 0.01,
    "gpt-4-32k": 0.02,
    "gpt-4-32k-0314": 0.02,
    "gpt-4-32k-0613": 0.02,
    "gpt-4-turbo": 0.003,
    "gpt-4o": 0.005,     
         
    # GPT-3.5 family
    "gpt-3.5-turbo": 0.0005,
    "gpt-3.5-turbo-0301": 0.0005,
    "gpt-3.5-turbo-0613": 0.0005,
    "gpt-3.5-turbo-1106": 0.0005,
    "gpt-3.5-turbo-0125": 0.0005,

    # Personalized alias
    "gpt-4.1": 0.002,
    "gpt-4.1 mini": 0.0004,
    "gpt-4.1 nano": 0.0001,
    "OpenAi o3": 0.002,
    "OpenAi o4-mini": 0.0011,
}

tokenizer_alias = {
    "gpt-4.1": "gpt-4",
    "gpt-4.1 mini": "gpt-4",
    "gpt-4.1 nano": "gpt-4",
    "OpenAi o3": "gpt-3.5-turbo",
    "OpenAi o4-mini": "gpt-4",
}

def calculate_token_cost(text, model):
    if model not in price_per_token:
        raise ValueError(f"Unsupported model: {model}")
    alias = tokenizer_alias.get(model, model)
    try:
        encoding = tiktoken.encoding_for_model(alias)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(text))
    unit_price = price_per_token[model]
    total_cost = token_count * unit_price
    print("Selected model:", model)
    print("Number of tokens:", token_count)
    print("Price per token ($):", round(unit_price, 8))
    print("Estimated total cost ($):", round(total_cost, 6))

if __name__ == "__main__":
    user_input = input("Enter your prompt:\n")
    model_name = input("Model name: ").strip()
    calculate_token_cost(user_input, model_name)
