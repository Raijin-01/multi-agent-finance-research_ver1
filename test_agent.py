from config.gemini import ask_gemini
from agents.loader import load_agent_prompt


agent_prompt = load_agent_prompt("market")

prompt = f"""
{agent_prompt}

Now respond to this test request:

Explain what a market research analyst should examine
when researching NVIDIA (NVDA).
"""

response = ask_gemini(prompt)

print("\n===== MARKET AGENT =====\n")
print(response)
