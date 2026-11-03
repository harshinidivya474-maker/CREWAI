# CREWAI – 10 Agent Projects

A FastAPI backend that exposes ten CrewAI demonstrations as HTTP API routes.

## Projects and routes

| Route | Project | Main concept |
|---|---|---|
| `POST /agent1` | Blog Post Crew | Sequential agents |
| `POST /agent2` | Competitor Analysis Crew | Hierarchical crew |
| `POST /agent3` | Meeting Notes Crew | Parallel analysis tasks |
| `POST /agent4` | Data Report Crew | Structured Pydantic output |
| `POST /agent5` | Customer Support | Request classification and routing |
| `POST /agent6` | Resume Screening | Job-role routing |
| `POST /agent7` | Bug Triage | Category/severity routing with a specialist crew |
| `POST /agent8` | Content Repurposing | Multiple platform-specific crews |
| `POST /agent9` | Lead Qualification | Hot/warm/cold lead routing |
| `POST /agent10` | Approval Report | Human review and feedback/revision |

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install packages:

   ```bash
   pip install -r req.txt
   ```

4. Set your Gemini key as an environment variable. Do not put the key in this repository.

   Windows PowerShell:

   ```powershell
   $env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   ```

   macOS/Linux:

   ```bash
   export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   ```

5. Start the server:

   ```bash
   uvicorn app:app --reload
   ```

6. Open `http://127.0.0.1:8000/docs` to test the endpoints.

## Render deployment

- Build Command: `pip install -r req.txt`
- Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Environment Variable: `GEMINI_API_KEY` = your Gemini API key
- Optional Environment Variable: `GEMINI_MODEL` = `gemini/gemini-3.5-flash-lite`

## Request examples

### Agent 1

```json
{
  "topic": "Applications of artificial intelligence in education"
}
```

### Agent 2

```json
{
  "urls": [
    "https://example.com",
    "https://example.org",
    "https://example.net"
  ]
}
```

### Agent 3

```json
{
  "transcript": "The team discussed the project. Asha will finish the login page by Friday."
}
```

### Agent 4

```json
{
  "csv_data": "Product,Quantity,Price,Region\nLaptop,2,50000,South\nMobile,3,20000,North"
}
```

### Agent 5

```json
{
  "customer_name": "Ravi",
  "request": "I cannot log in to my account."
}
```

### Agent 6

```json
{
  "candidate_name": "Candidate A",
  "job_role": "Python Developer",
  "resume_text": "Python, FastAPI, SQL, BCA, internship project experience..."
}
```

### Agent 7

```json
{
  "bug_title": "Login button not working",
  "bug_description": "Clicking the login button does not submit the form."
}
```

### Agent 8

```json
{
  "topic": "Cybersecurity awareness",
  "original_content": "Use strong passwords and enable multi-factor authentication."
}
```

### Agent 9

```json
{
  "customer_name": "Rahul",
  "company_name": "ABC Technologies",
  "requirement": "CRM solution",
  "budget": 100000,
  "timeline_days": 15
}
```

### Agent 10

First call to create a draft:

```json
{
  "project_title": "Student Management System",
  "project_details": "A web application to manage student records, attendance and results."
}
```

To request revision, send the previous `draft_report` as `previous_draft` and include `feedback`. To mark a report approved, set `approved` to `true`.

## Notes

- The API key is read from the `GEMINI_API_KEY` environment variable.
- Never commit API keys, passwords or secrets to GitHub.
- `/docs` provides interactive API testing.
- Agent 2 retrieves public website pages; some sites may block automated requests.
- Agent 4 expects a CSV with a `Sales` column or both `Quantity` and `Price`.
- Agent 10 models human approval through a second API request because a deployed API cannot wait for an interactive terminal `input()` prompt.
