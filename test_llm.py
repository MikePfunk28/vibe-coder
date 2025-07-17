from utils.call_llm import call_llm

print("Testing LLM call...")
response = call_llm('Hello! Can you respond with just "Working!"?')
print(f'Response: {response}')
