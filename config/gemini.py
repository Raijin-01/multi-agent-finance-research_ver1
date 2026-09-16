import os
from google import genai


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Set it in your environment before running the team."
        )

    return genai.Client(api_key=api_key)


def ask_gemini(prompt: str) -> str:
    client = get_client()

    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    if not response.output_text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.output_text
