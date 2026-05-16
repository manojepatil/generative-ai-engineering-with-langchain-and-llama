import os
from datetime import datetime

# Mock some essential LangChain components for demonstration
class MockLLM:
    def __init__(self, model_name="MockGPT-3", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.call_count = 0

    def invoke(self, prompt, stop=None):
        self.call_count += 1
        # Simulate LLM response based on prompt content
        if "current date" in prompt.lower():
            return f"The current date is {datetime.now().strftime('%Y-%m-%d')}."
        elif "calculate" in prompt.lower() or "sum" in prompt.lower():
            try:
                numbers = [int(s) for s in prompt.split() if s.isdigit()]
                return f"The sum of the numbers is {sum(numbers)}."
            except ValueError:
                return "I couldn't calculate that. Please provide valid numbers."
        else:
            return f"MockLLM '{self.model_name}' processed: \"{prompt}\""

class PromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs):
        formatted_template = self.template
        for var in self.input_variables:
            if var not in kwargs:
                raise ValueError(f"Missing input variable: {var}")
            formatted_template = formatted_template.replace("{" + var + "}", str(kwargs[var]))
        return formatted_template

class LLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def invoke(self, inputs):
        formatted_prompt = self.prompt.format(**inputs)
        return self.llm.invoke(formatted_prompt)

# 1. Define an LLM (simulated)
mock_llm = MockLLM()

# 2. Define a Prompt Template
greeting_template = PromptTemplate(
    template="Hello, my name is {name} and I am {age} years old. Tell me an interesting fact about my age.",
    input_variables=["name", "age"]
)

# 3. Create a Chain: Orchestrating an LLM call with a prompt
# This chain takes name and age, formats them into the prompt, and sends to the LLM.
greeting_chain = LLMChain(llm=mock_llm, prompt=greeting_template)

# Invoke the chain with inputs
print("--- Greeting Chain Output ---")
result1 = greeting_chain.invoke({"name": "Alice", "age": 30})
print(f"Result: {result1}")
print(f"LLM Call Count: {mock_llm.call_count}\n")


# 4. Agent-like behavior: Illustrating conditional LLM calls based on task
# While not a full LangChain Agent, this shows conditional logic around LLM calls.
def simple_agent_flow(query, llm_instance):
    if "date" in query.lower():
        prompt = PromptTemplate(template="What is the current date?", input_variables=[])
        chain = LLMChain(llm=llm_instance, prompt=prompt)
        return chain.invoke({})
    elif "sum" in query.lower():
        prompt = PromptTemplate(template=f"Calculate the sum of numbers in this query: {query}", input_variables=[])
        chain = LLMChain(llm=llm_instance, prompt=prompt)
        return chain.invoke({})
    else:
        prompt = PromptTemplate(template=f"Respond to the following: {query}", input_variables=[])
        chain = LLMChain(llm=llm_instance, prompt=prompt)
        return chain.invoke({})

print("--- Agent-like Flow Output ---")
result2 = simple_agent_flow("What is the current date?", mock_llm)
print(f"Query: 'What is the current date?'\nResult: {result2}")

result3 = simple_agent_flow("Can you sum 10 and 20?", mock_llm)
print(f"Query: 'Can you sum 10 and 20?'\nResult: {result3}")

result4 = simple_agent_flow("Tell me a fun fact.", mock_llm)
print(f"Query: 'Tell me a fun fact.'\nResult: {result4}")

print(f"\nTotal LLM Call Count after Agent-like flow: {mock_llm.call_count}")
