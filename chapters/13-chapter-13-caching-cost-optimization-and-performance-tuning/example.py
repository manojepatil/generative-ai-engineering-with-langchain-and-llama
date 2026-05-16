# caching_example.py

from functools import lru_cache
import time

class LLMSimulator:
    """A simulated LLM to demonstrate caching."""
    def __init__(self, processing_time=0.5):
        self.processing_time = processing_time
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Simulates an LLM call with a delay."""
        self.call_count += 1
        time.sleep(self.processing_time)  # Simulate network latency and processing
        response = f"Simulated response to: '{prompt}' (call {self.call_count})"
        return response

# Initialize our simulated LLM
llm_no_cache = LLMSimulator()

# Initialize our simulated LLM with caching applied using lru_cache
# maxsize=128 means it will cache up to 128 unique prompts
@lru_cache(maxsize=128)
def cached_llm_generate(prompt: str, simulator: LLMSimulator) -> str:
    """A cached version of the LLM generate function."""
    return simulator.generate(prompt)

# Create another simulator instance for the cached calls to track its specific calls
llm_cached_simulator = LLMSimulator()

print("--- Demonstrating LLM without cache ---")
start_time = time.time()
print(f"Call 1: {llm_no_cache.generate('Tell me a joke.')}")
print(f"Call 2: {llm_no_cache.generate('Tell me a joke.')}") # Same prompt, still takes time
end_time = time.time()
print(f"Time taken without cache: {end_time - start_time:.2f} seconds")
print(f"Total LLM calls without cache: {llm_no_cache.call_count}\n")

print("--- Demonstrating LLM with simple LRU cache ---")
start_time = time.time()
print(f"Cached Call 1: {cached_llm_generate('Tell me a story.', llm_cached_simulator)}")
# The second call with the same prompt should be significantly faster
print(f"Cached Call 2: {cached_llm_generate('Tell me a story.', llm_cached_simulator)}")
print(f"Cached Call 3: {cached_llm_generate('Tell me a different story.', llm_cached_simulator)}")
print(f"Cached Call 4: {cached_llm_generate('Tell me a story.', llm_cached_simulator)}") # Cache hit
end_time = time.time()
print(f"Time taken with cache: {end_time - start_time:.2f} seconds")
# Notice that the underlying simulator's call count only increments for unique prompts
print(f"Total actual LLM calls (simulator behind cache): {llm_cached_simulator.call_count}")

# Clear the cache for clean re-runs in a real application, if needed
cached_llm_generate.cache_clear()
