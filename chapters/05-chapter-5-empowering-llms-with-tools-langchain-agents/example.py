from typing import List, Union
from langchain_core.agents import AgentAction, AgentFinish, AgentExecutor
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable

# --- Mock LLM Component ---
# In a real scenario, this would be a call to a model like OpenAI's GPT.
# Here, we simulate simple responses for demonstration.
def mock_llm_chain(messages: List[BaseMessage]) -> AIMessage:
    last_human_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""

    if "current time" in last_human_message.lower():
        # Simulate a thought process to use a tool
        return AIMessage(content="Thought: The user is asking for the current time. I should use the 'get_current_time' tool.\nAction: get_current_time\nAction Input: {}")
    elif "hello" in last_human_message.lower():
        return AIMessage(content="Hello there! How can I assist you today?")
    elif "tool_output" in last_human_message.lower():
        # Simulate a follow-up after a tool use
        return AIMessage(content="Thought: I have successfully retrieved the current time. I should now respond to the user.\nFinal Answer: The current time is 10:30 AM (simulated).")
    else:
        return AIMessage(content="Thought: I don't need to use a tool for this. I can answer directly.\nFinal Answer: I can answer general questions.")


# --- Mock Tool Component ---
# Tools are functions that the agent can call.
def get_current_time() -> str:
    """Returns the current simulated time."""
    return "10:30 AM (simulated)"

# Define our tools with descriptions for the LLM to understand.
tools = [
    {
        "name": "get_current_time",
        "description": "Useful for when you need to know the current time.",
        "func": get_current_time
    }
]

# --- Agent Core Logic ---
# This class acts as a simple agent executor.
class SimpleAgentExecutor(Runnable):
    def __init__(self, llm_chain, tools):
        self.llm_chain = llm_chain
        self.tools_map = {tool['name']: tool['func'] for tool in tools}

    def _parse_llm_output(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(return_values={"output": text.split("Final Answer:")[-1].strip()}, log=text)
        elif "Action:" in text and "Action Input:" in text:
            import re
            action_match = re.search(r"Action: (.+?)\nAction Input: (.+)", text)
            if action_match:
                return AgentAction(tool=action_match.group(1).strip(), tool_input=action_match.group(2).strip(), log=text)
        raise OutputParserException(f"Could not parse LLM output: {text}")

    def invoke(self, input: str, **kwargs) -> dict:
        intermediate_steps = []
        full_log = []
        messages: List[BaseMessage] = [HumanMessage(content=input)]

        while True:
            llm_response_message = self.llm_chain(messages)
            llm_output = llm_response_message.content
            full_log.append(llm_output)

            try:
                action_or_finish = self._parse_llm_output(llm_output)
            except OutputParserException as e:
                # If LLM produces unparseable output, try to guide it.
                messages.append(AIMessage(content=llm_output))
                messages.append(HumanMessage(content=f"Error parsing your last response: {e}. Please ensure you follow the 'Action: <tool_name>\\nAction Input: <input>' or 'Final Answer: <answer>' format."))
                continue

            if isinstance(action_or_finish, AgentFinish):
                return {"output": action_or_finish.return_values["output"], "log": "\n---\n".join(full_log)}
            elif isinstance(action_or_finish, AgentAction):
                tool_name = action_or_finish.tool
                tool_input = action_or_finish.tool_input

                if tool_name in self.tools_map:
                    print(f"Agent uses tool: {tool_name} with input: {tool_input}")
                    tool_output = self.tools_map[tool_name]() # Call the tool
                    print(f"Tool output: {tool_output}")
                    intermediate_steps.append((action_or_finish, tool_output))
                    messages.append(AIMessage(content=llm_output)) # Add LLM's action message
                    messages.append(HumanMessage(content=f"tool_output:{tool_output}")) # Add tool output as new message
                else:
                    return {"output": f"Error: Unknown tool {tool_name}", "log": "\n---\n".join(full_log)}

# Instantiate the agent
agent_executor = SimpleAgentExecutor(llm_chain=mock_llm_chain, tools=tools)

# Run the agent with a query that requires a tool
print("--- Agent Query 1: What is the current time? ---")
result1 = agent_executor.invoke("What is the current time?")
print("Final Agent Output:", result1["output"])
print("\n--- Full Log ---")
print(result1["log"])

print("\n--- Agent Query 2: Say hello ---")
result2 = agent_executor.invoke("Say hello")
print("Final Agent Output:", result2["output"])
print("\n--- Full Log ---")
print(result2["log"])
