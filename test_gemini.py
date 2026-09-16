import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

response = client.interactions.create(
    model="gemini-3.6-flash",
    input="Say: Gemini is connected successfully."
)

print(response.output_text)
