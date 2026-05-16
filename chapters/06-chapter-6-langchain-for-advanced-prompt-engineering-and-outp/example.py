from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.prompts.example_selector import LengthBasedExampleSelector
from langchain_community.llms import FakeListLLM  # Simulate LLM for testing
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# 1. Define a Pydantic model for structured output parsing
class ProductInfo(BaseModel):
    product_name: str = Field(description="The name of the product")
    category: str = Field(description="The category of the product (e.g., Electronics, Clothing)")
    price: float = Field(description="The price of the product in USD")

# Create a parser instance
parser = PydanticOutputParser(pydantic_object=ProductInfo)

# 2. Define few-shot examples for the LLM
examples = [
    {
        "input": "I need a gadget for making coffee.",
        "output": '{"product_name": "Coffee Maker", "category": "Kitchen Appliances", "price": 49.99}'
    },
    {
        "input": "Tell me about a waterproof watch.",
        "output": '{"product_name": "Smart Watch", "category": "Wearable Technology", "price": 199.99}'
    },
    {
        "input": "Where can I find shirts?",
        "output": '{"product_name": "T-Shirt", "category": "Apparel", "price": 25.00}'
    },
]

# 3. Create a sophisticated prompt template with few-shot examples and output parsing instructions
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# Use LengthBasedExampleSelector to dynamically choose examples
# This ensures that shorter inputs get fewer examples, preventing prompt overflow
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50,  # Max length of combined examples to include in the prompt
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Extract product information from the user's request as a JSON object, following the schema below.\n"
           "Schema:\n{format_instructions}\n",
    suffix="Input: {user_request}\nOutput:",
    input_variables=["user_request"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. Simulate an LLM - in a real scenario, this would be OpenAI, Cohere, etc.
# The FakeListLLM returns predefined responses, simulating the LLM generating the structured JSON.
llm = FakeListLLM(responses=[
    '{"product_name": "Wireless Headphones", "category": "Electronics", "price": 120.00}',
    '{"product_name": "Running Shoes", "category": "Footwear", "price": 85.50}'
])

# 5. Generate and parse output for a user query
user_query_1 = "I want some good headphones for music."
formatted_prompt_1 = few_shot_prompt.format(user_request=user_query_1)
print(f"--- Prompt 1 ---")
print(formatted_prompt_1)
llm_output_1 = llm.invoke(formatted_prompt_1)
print(f"--- LLM Raw Output 1 ---")
print(llm_output_1)
parsed_output_1 = parser.parse(llm_output_1)
print(f"--- Parsed Product 1 ---")
print(parsed_output_1.category) # Accessing structured data
print(f"Product Name: {parsed_output_1.product_name}, Price: ${parsed_output_1.price}\n")

user_query_2 = "Looking for new trainers." # A shorter query, might trigger fewer examples
formatted_prompt_2 = few_shot_prompt.format(user_request=user_query_2)
print(f"--- Prompt 2 ---")
print(formatted_prompt_2)
llm_output_2 = llm.invoke(formatted_prompt_2)
print(f"--- LLM Raw Output 2 ---")
print(llm_output_2)
parsed_output_2 = parser.parse(llm_output_2)
print(f"--- Parsed Product 2 ---")
print(f"Product Name: {parsed_output_2.product_name}, Category: {parsed_output_2.category}, Price: ${parsed_output_2.price}")
