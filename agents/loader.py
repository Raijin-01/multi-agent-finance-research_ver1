from pathlib import Path


AGENTS_DIR = Path(__file__).resolve().parent


def load_agent_prompt(agent_name: str) -> str:
    path = AGENTS_DIR / f"{agent_name}.md"

    if not path.exists():
        raise FileNotFoundError(f"Agent prompt not found: {path}")

    return path.read_text(encoding="utf-8")
