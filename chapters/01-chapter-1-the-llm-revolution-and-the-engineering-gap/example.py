import time

# Simulate a simple LLM call (e.g., a function that generates text)
def simple_llm_call(prompt: str) -> str:
    """
    A placeholder for an actual LLM API call.
    Simulates a small delay and returns a fixed response.
    """
    time.sleep(0.1)  # Simulate API latency
    if "hello" in prompt.lower():
        return "Hello there! How can I help you today?"
    elif "weather" in prompt.lower():
        return "I can't check the weather, as I am an AI model."
    else:
        return "I received your request."

# --- The "engineering gap" in action ---

# Problem 1: Hardcoding logic around LLM calls
def get_llm_response_hardcoded(user_query: str) -> str:
    print(f"User query: '{user_query}'")
    llm_output = simple_llm_call(user_query)
    print(f"Raw LLM output: '{llm_output}'")

    # Hardcoded post-processing logic
    if "hello" in llm_output.lower():
        return "Greeting received: " + llm_output
    elif "weather" in llm_output.lower():
        return "Apology response based on LLM output: " + llm_output
    else:
        return "Default processing: " + llm_output

# Problem 2: Lack of observability (e.g., timing, retries, structured logging)
def get_llm_response_unstructured(user_query: str) -> str:
    start_time = time.time()
    response = simple_llm_call(user_query)
    end_time = time.time()
    print(f"LLM call took {end_time - start_time:.2f} seconds.")
    # More complex scenarios would involve error handling, retries, etc.
    return response

# Demonstrate the "engineering gap"
if __name__ == "__main__":
    print("--- Demonstrating Hardcoded Logic ---")
    print(get_llm_response_hardcoded("Hello, LLM!"))
    print(get_llm_response_hardcoded("What's the weather like?"))
    print(get_llm_response_hardcoded("Tell me a story."))

    print("\n--- Demonstrating Lack of Observability ---")
    # In a real system, we'd want structured logs, metrics, etc.
    # This simple print statement is a bare minimum for illustrating the point.
    print(get_llm_response_unstructured("Please summarize this text."))
    print(get_llm_response_unstructured("Another quick request."))

    print("\nImagine scaling this with multiple LLM calls, tools, data sources, "
          "and complex conversational state management. The current approach "
          "quickly becomes unmanageable and unmaintainable.")
    print("This is the 'engineering gap' that frameworks like LangChain "
          "and LlamaIndex aim to bridge by providing structured abstractions.")
