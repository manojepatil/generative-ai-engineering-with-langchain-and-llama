import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import FakeListLLM  # Use FakeListLLM for no external services

# Define a simple LLMChain
# This chain takes a product name and generates a catchy slogan.
llm_slogan = FakeListLLM(responses=["Catchy slogan for {product_name}: Innovate. Inspire. Impact.",
                                     "Catchy slogan for {product_name}: The Future is Here."])
prompt_slogan = PromptTemplate.from_template("Generate a catchy slogan for a product named '{product_name}'.")
slogan_chain = prompt_slogan | llm_slogan | StrOutputParser()

# Define another simple LLMChain
# This chain expands on a catchy slogan to explain its meaning.
llm_explanation = FakeListLLM(responses=["Explanation of '{slogan}': This slogan emphasizes innovation, inspiration, and positive impact, highlighting the product's core values.",
                                        "Explanation of '{slogan}': It suggests advanced technology and forward-thinking design."])
prompt_explanation = PromptTemplate.from_template("Explain the meaning and intent behind the slogan: '{slogan}'.")
explanation_chain = prompt_explanation | llm_explanation | StrOutputParser()

# Combine the two chains into a SequentialChain
# The output of the first chain (slogan) becomes the input for the second chain.
from langchain.chains import SequentialChain

# Define the overall input variables for the SequentialChain
overall_input_variables = ["product_name"]

# Define the chains in their desired order
# Each chain's output must map correctly to the next chain's input.
chained_operations = [
    {
        "name": "Slogan Generator",
        "chain": slogan_chain,
        "input_variables": ["product_name"],
        "output_variables": ["slogan"] # Name the output for subsequent chains
    },
    {
        "name": "Slogan Explanation",
        "chain": explanation_chain,
        "input_variables": ["slogan"],
        "output_variables": ["explanation"] # Name the output for the final result
    }
]

# Instantiate SequentialChain
# `input_variables` are what the user provides to the whole chain.
# `output_variables` are what we want to get back from the whole chain.
product_idea_chain = SequentialChain(
    chains=[slogan_chain.with_config(run_name="slogan_gen"),  # Use with_config for explicit run names
            explanation_chain.with_config(run_name="slogan_explain")],
    input_variables=["product_name"],
    output_variables=["slogan", "explanation"],
    verbose=False  # Set to True to see intermediate steps
)

# Invoke the SequentialChain with an initial input
result = product_idea_chain.invoke({"product_name": "QuantuMind AI"})

# Print the final result, demonstrating how multiple steps were executed
print(f"Product Name: QuantuMind AI")
print(f"Generated Slogan: {result['slogan'].strip()}")
print(f"Slogan Explanation: {result['explanation'].strip()}")

# Test with another product
result2 = product_idea_chain.invoke({"product_name": "EcoCharge Battery"})
print("\n--- Another Example ---")
print(f"Product Name: EcoCharge Battery")
print(f"Generated Slogan: {result2['slogan'].strip()}")
print(f"Slogan Explanation: {result2['explanation'].strip()}")
