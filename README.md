# AI Finance Research Team

A data-grounded multi-agent financial research system built with **Google Gemini** and **LangGraph**.

## Overview

The system separates data retrieval from LLM interpretation. Market data and fundamentals are retrieved with `yfinance`, recent news is collected from RSS feeds, and technical indicators are calculated in Python with pandas/numpy. Gemini is used to interpret the retrieved information rather than inventing financial figures.

## Architecture

```text
                         MANAGER
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
        MARKET_DATA      NEWS_DATA    FUNDAMENTALS_DATA
             |              |              |
             v              v              v
          MARKET          NEWS       FUNDAMENTALS
             \              |              /
              +-------------+-------------+
                            |
                            v
                     TECHNICAL_DATA
                            |
                            v
                        TECHNICAL
                            |
                            v
                           RISK
                            |
                            v
                       VERIFICATION
                            |
                            v
                          CRITIC
                            |
                            v
                          REPORT
                            |
                            v
                           END
```

## Features

- Multi-agent orchestration with LangGraph
- Google Gemini API for analysis and report generation
- Real market and financial data through Yahoo Finance/yfinance
- Recent news retrieval through RSS feeds
- Python-calculated SMA, EMA, RSI, MACD, ATR, Bollinger Bands, support/resistance and pivot levels
- Explicit `DATA NOT AVAILABLE` handling instead of fabricated values
- Source and retrieval timestamps
- Automated freshness and consistency verification
- Critic/QC pass before the final report
- Markdown report export
- Best-effort PDF export
- Retry with exponential backoff
- In-process TTL caching
- Graceful failure handling for non-critical nodes
- Designed for Google Colab and does not require a GPU

## Project Structure

```text
multi-agent-finance-research_ver1/
├── finance_research_team.py
├── README.md
├── requirements.txt
├── .gitignore
├── reports/
│   └── .gitkeep
└── notebooks/
    └── .gitkeep
```

## Requirements

- Python 3.10+
- Google Colab or a local Python environment
- Gemini API access
- Internet access for market/news retrieval

No GPU is required because model inference is performed through the Gemini API.

## Installation

```bash
pip install -r requirements.txt
```

For Google Colab, the main Python file also contains dependency bootstrap logic for the required packages.

## Gemini Setup

The script expects a global `client` object to already exist. In Colab, initialize your Gemini client using your `GEMINI_API_KEY` secret before running the research system.

The expected invocation is:

```python
response = client.interactions.create(
    model="gemini-3.6-flash",
    input="your prompt"
)
print(response.output_text)
```

Do **not** commit API keys or secrets to this repository.

## Run

The default example runs an NVIDIA research workflow:

```python
RUN_EXAMPLE = True
```

To run your own company:

```python
result = run_research(
    company_name="Apple Inc.",
    ticker="AAPL",
    research_question="Conduct a comprehensive financial analysis covering market position, recent news, fundamentals, technical setup, and major risks."
)
```

## Data Grounding

The pipeline is designed so that retrieved data and calculated indicators are passed to Gemini for interpretation. Missing values are explicitly represented as `DATA NOT AVAILABLE`. The report instructions also require numeric tables to identify their source/date and require scenario figures to be labeled as model assumptions.

## Important Limitation

Yahoo Finance/yfinance is used as the primary automated market and fundamental data source in this version. For production-grade investment research, primary sources such as company filings, earnings releases, investor-relations material, and regulatory filings should be added and cross-checked.

## Disclaimer

This project is for research, education, and software-development purposes. It does not provide personalized investment advice or a buy/sell/hold recommendation.

## License

Add a license before distributing the project publicly if you want to specify reuse terms.
