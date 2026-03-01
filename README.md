# NewsSummarizer

NewsSummarizer reads article URLs from an Excel file, scrapes the article content, generates a consolidated AI news brief using a local Ollama model, and emails the final summary.

## Features

- Reads news links from `NewsLinks.xlsx`
- Scrapes article content using `crawl4ai`
- Summarizes all stories via local Ollama model (`llama3.2:3b` by default)
- Produces a markdown-style newsletter summary with inline source links
- Sends the summary by email (plain text + HTML)
- Falls back to printing summary in console if email credentials are missing

## Project Requirements

### System

- Python 3.10+
- Ollama installed and running locally
- Chromium browser support for crawler runtime (`playwright install`)

### Python Packages

Install required packages:

```bash
pip install langgraph langchain_ollama langchain-core crawl4ai pandas openpyxl python-dotenv
```

Install Playwright browser dependencies:

```bash
playwright install
```

## Ollama Model Setup

Start Ollama and pull the model used by default:

```bash
ollama pull llama3.2:3b
```

You can override the model with `OLLAMA_MODEL` in `.env`.

## Configuration

Create a `.env` file in project root:

```env
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
RECIPIENT_EMAIL=recipient@gmail.com
OLLAMA_MODEL=llama3.2:3b
```

Notes:

- `EMAIL_PASSWORD` should be a Gmail App Password, not your account password.
- If email variables are missing, the script prints the summary to console instead of sending mail.

## Input File Format

The script expects an Excel file named `NewsLinks.xlsx` in project root with a column named exactly:

- `URL`

Example:

| URL |
| --- |
| https://example.com/article-1 |
| https://example.com/article-2 |

## How It Works (End-to-End)

Pipeline implemented in `Agent.py`:

1. Read URLs from `NewsLinks.xlsx` (`read_excel_node`)
2. Scrape each URL and extract text (`scrape_links_node`)
3. Summarize all scraped content with Ollama (`summarize_node`)
4. Email summary as newsletter (`send_email_node`)

The flow is orchestrated using LangGraph:

`START -> read_excel -> scrape -> summarize -> email -> END`

## Running the Project

From project root:

```bash
python Agent.py
```

## Output

- Success path: summary is emailed to `RECIPIENT_EMAIL`
- Partial scrape failures: failed URLs are skipped; successful articles are still summarized
- No scrape success: execution stops with error
- Email send failure: summary is printed in terminal

## Troubleshooting

- `ModuleNotFoundError`: install missing package(s) with `pip install ...`
- Playwright/browser issues: run `playwright install`
- Ollama connection/model errors:
  - ensure Ollama daemon is running
  - ensure model exists (`ollama pull llama3.2:3b`)
- Excel errors:
  - verify file is named `NewsLinks.xlsx`
  - verify `URL` column exists and contains valid links
- Gmail auth errors:
  - confirm App Password is used
  - confirm account allows app password usage

## Security Recommendations

- Do not commit `.env` to source control
- Rotate credentials immediately if exposed
- Use a dedicated sender mailbox for automation


----------

pip install -r requirements.txt

uvicorn app:app --reload   