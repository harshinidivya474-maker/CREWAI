import os
import io
import re
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM, Process
from urllib.parse import urlparse

app = FastAPI(
    title="CREWAI - 10 Agent Projects",
    description="FastAPI backend for ten CrewAI demonstrations.",
    version="1.1.0",
)

# Teacher's required common input format:
# {"input": "your text here"}
class Query(BaseModel):
    input: str = Field(min_length=2, max_length=100000)


def get_llm():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server."
        )
    return LLM(
        model=os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite"),
        api_key=key,
        temperature=0.3,
    )


def run_crew(agent_specs, task_specs, process=Process.sequential, manager=False):
    llm = get_llm()
    agents = [
        Agent(
            role=spec["role"],
            goal=spec["goal"],
            backstory=spec["backstory"],
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=4,
            tools=spec.get("tools", []),
        )
        for spec in agent_specs
    ]

    tasks = []
    for spec in task_specs:
        tasks.append(
            Task(
                description=spec["description"],
                expected_output=spec["expected_output"],
                agent=agents[spec.get("agent", 0)],
                context=[tasks[j] for j in spec.get("context", [])],
            )
        )

    crew_kwargs = {
        "agents": agents,
        "tasks": tasks,
        "process": process,
        "verbose": False,
    }
    if manager:
        crew_kwargs["manager_llm"] = llm

    result = Crew(**crew_kwargs).kickoff()
    return str(result)


def one_agent_result(role, goal, backstory, prompt):
    return run_crew(
        [{"role": role, "goal": goal, "backstory": backstory}],
        [{
            "description": prompt,
            "expected_output": "A clear, complete and useful response.",
        }],
    )


def get_urls(text: str) -> List[str]:
    urls = re.findall(r"https?://[^\s,]+", text)
    # Remove common trailing punctuation from copied URLs
    return [u.rstrip(").,;]") for u in urls]


def website_text(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=422, detail=f"Invalid URL: {url}")
    try:
        response = requests.get(
            url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "noscript"]):
            element.decompose()
        return soup.get_text(" ", strip=True)[:7000]
    except requests.RequestException as exc:
        return f"Could not retrieve {url}: {exc}"


# -------------------- Agent 1: Blog Post Crew --------------------
@app.post("/agent1", tags=["Agent 1 - Blog Post Crew"])
def agent1(data: Query):
    result = one_agent_result(
        "Professional Blog Writer",
        "Write an informative, engaging and accurate blog post.",
        "You are an experienced blog writer and editor.",
        f"Write an approximately 600-word blog post about:\n{data.input}\n"
        "Include a title, introduction, headings and conclusion. "
        "Do not invent facts or sources.",
    )
    return {"output": result}


# -------------------- Agent 2: Competitor Analysis --------------------
@app.post("/agent2", tags=["Agent 2 - Competitor Analysis"])
def agent2(data: Query):
    urls = get_urls(data.input)
    if len(urls) != 3:
        raise HTTPException(
            status_code=422,
            detail="Enter exactly three competitor website URLs in the input text.",
        )
    if len(set(urls)) != 3:
        raise HTTPException(
            status_code=422,
            detail="Provide three different competitor URLs.",
        )

    website_extracts = {url: website_text(url) for url in urls}
    supplied = "\n\n".join(
        f"WEBSITE: {url}\nCONTENT: {content}"
        for url, content in website_extracts.items()
    )

    report = run_crew(
        [
            {
                "role": "Competitor Analysis Manager",
                "goal": "Coordinate an evidence-based competitor analysis.",
                "backstory": "You are a strategic market analyst.",
            },
            {
                "role": "Market Research Analyst",
                "goal": "Extract comparable business facts from the supplied sites.",
                "backstory": "You identify products, services, audiences and differentiators.",
            },
            {
                "role": "Business Strategist",
                "goal": "Prepare practical recommendations based on evidence.",
                "backstory": "You turn verified findings into useful business strategy.",
            },
        ],
        [
            {
                "description": (
                    "Analyze these website extracts and prepare comparison notes. "
                    "Do not invent facts:\n" + supplied
                ),
                "expected_output": "Evidence-based competitor research notes.",
                "agent": 1,
            },
            {
                "description": (
                    "Create a competitor report with summary, profiles, "
                    "products/services, pricing (write Not found if absent), "
                    "SWOT, opportunities, recommendations and source URLs."
                ),
                "expected_output": "Complete competitor analysis report.",
                "agent": 2,
                "context": [0],
            },
        ],
        process=Process.hierarchical,
        manager=True,
    )
    return {"output": report}


# -------------------- Agent 3: Meeting Notes --------------------
@app.post("/agent3", tags=["Agent 3 - Meeting Notes"])
def agent3(data: Query):
    transcript = data.input
    llm = get_llm()
    agent_specs = [
        {
            "role": "Meeting Summarizer",
            "goal": "Summarize the meeting in five concise points.",
            "backstory": "You create accurate meeting summaries.",
        },
        {
            "role": "Action Item Analyst",
            "goal": "Extract action items, owners and due dates.",
            "backstory": "You identify commitments and deadlines.",
        },
        {
            "role": "Decision Logger",
            "goal": "Identify decisions made.",
            "backstory": "You maintain accurate decision records.",
        },
        {
            "role": "Meeting Notes Organizer",
            "goal": "Combine findings into polished meeting notes.",
            "backstory": "You create clear professional meeting records.",
        },
    ]
    agents = [
        Agent(
            role=s["role"], goal=s["goal"], backstory=s["backstory"],
            llm=llm, verbose=False
        )
        for s in agent_specs
    ]
    t1 = Task(
        description="Summarize this transcript in exactly five concise points:\n" + transcript,
        expected_output="Five-point meeting summary.",
        agent=agents[0],
    )
    t2 = Task(
        description="Extract action items, owners and due dates. Use 'Not specified' if absent:\n" + transcript,
        expected_output="Action list with owners and due dates.",
        agent=agents[1],
    )
    t3 = Task(
        description="List only decisions explicitly supported by this transcript:\n" + transcript,
        expected_output="List of meeting decisions.",
        agent=agents[2],
    )
    t4 = Task(
        description="Combine the previous outputs into Meeting Summary, Action Items and Decisions Made.",
        expected_output="Complete formatted meeting notes.",
        agent=agents[3],
        context=[t1, t2, t3],
    )
    result = Crew(
        agents=agents, tasks=[t1, t2, t3, t4],
        process=Process.sequential, verbose=False
    ).kickoff()
    return {"output": str(result)}


# -------------------- Agent 4: Sales Analysis --------------------
@app.post("/agent4", tags=["Agent 4 - Sales Analysis"])
def agent4(data: Query):
    try:
        df = pd.read_csv(io.StringIO(data.input))
        if "Sales" not in df.columns:
            if {"Quantity", "Price"}.issubset(df.columns):
                df["Sales"] = (
                    pd.to_numeric(df["Quantity"], errors="coerce")
                    * pd.to_numeric(df["Price"], errors="coerce")
                )
            else:
                raise HTTPException(
                    status_code=422,
                    detail="CSV must contain a Sales column or both Quantity and Price columns.",
                )

        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
        df = df.dropna(subset=["Sales"])
        if df.empty:
            raise HTTPException(status_code=422, detail="No valid sales records found.")

        top_products = []
        if "Product" in df.columns:
            top_products = (
                df.groupby("Product")["Sales"]
                .sum().sort_values(ascending=False).head(3).index.astype(str).tolist()
            )

        stats = {
            "total_rows": int(len(df)),
            "total_sales": float(df["Sales"].sum()),
            "average_sales": float(df["Sales"].mean()),
            "top_products": top_products,
        }
    except pd.errors.ParserError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {exc}")

    report = one_agent_result(
        "Sales Report Reviewer",
        "Explain sales statistics clearly without changing the figures.",
        "You are a careful business data reviewer.",
        "Analyze these verified sales statistics and explain key findings:\n"
        + str(stats),
    )
    return {"output": f"Sales statistics:\n{stats}\n\nAnalysis:\n{report}"}


# -------------------- Agent 5: Customer Support --------------------
@app.post("/agent5", tags=["Agent 5 - Customer Support"])
def agent5(data: Query):
    text = data.input.lower()
    technical_words = [
        "error", "bug", "crash", "login", "password", "not working",
        "technical", "website", "app", "network",
    ]
    billing_words = [
        "payment", "bill", "billing", "refund", "invoice", "charged",
        "subscription", "price", "transaction", "money",
    ]
    category = (
        "technical" if any(word in text for word in technical_words)
        else "billing" if any(word in text for word in billing_words)
        else "general"
    )
    response = one_agent_result(
        f"{category.title()} Support Specialist",
        "Give a polite, practical and accurate customer support response.",
        f"You handle {category} customer enquiries. Never invent company policies.",
        "Customer support request:\n" + data.input,
    )
    return {"output": response}


# -------------------- Agent 6: Resume Screening --------------------
@app.post("/agent6", tags=["Agent 6 - Resume Screening"])
def agent6(data: Query):
    text = data.input.lower()
    role_category = (
        "Python" if "python" in text
        else "Web Development" if any(x in text for x in ["web", "frontend", "react"])
        else "Data" if any(x in text for x in ["data", "analyst", "analytics"])
        else "General"
    )
    report = one_agent_result(
        f"{role_category} Resume Screener",
        "Evaluate job-related qualifications and explain strengths and gaps.",
        "You are a fair recruiter. Assess only job-related evidence.",
        "Screen the following resume and job information. Identify relevant skills, "
        "strengths, gaps and a recommendation for further review:\n" + data.input,
    )
    return {"output": report}


# -------------------- Agent 7: Bug Triage --------------------
@app.post("/agent7", tags=["Agent 7 - Bug Triage"])
def agent7(data: Query):
    text = data.input.lower()
    category = (
        "frontend" if any(x in text for x in ["ui", "css", "button", "layout", "frontend", "display"])
        else "backend" if any(x in text for x in ["api", "database", "server", "backend", "sql"])
        else "general"
    )
    severity = (
        "critical" if any(x in text for x in ["crash", "data loss", "security breach", "system down"])
        else "high" if any(x in text for x in ["blocking", "cannot access", "not working", "failure"])
        else "medium" if any(x in text for x in ["slow", "minor", "incorrect"])
        else "low"
    )
    report = one_agent_result(
        f"{category.title()} Bug Analyst",
        "Prepare an evidence-aware bug triage report and debugging recommendations.",
        "You are a software debugging specialist. Do not claim to have reproduced or fixed the issue.",
        f"Bug report:\n{data.input}\nInitial category: {category}\nInitial severity: {severity}\n"
        "Explain possible causes, affected component, debugging steps, possible solutions "
        "and user impact. Separate facts from hypotheses.",
    )
    return {"output": report}


# -------------------- Agent 8: Content Repurposing --------------------
@app.post("/agent8", tags=["Agent 8 - Content Repurposing"])
def agent8(data: Query):
    platforms = [
        ("LinkedIn", "Create a professional post with a strong opening and CTA."),
        ("Instagram", "Create an engaging caption with emojis, CTA and 5-8 hashtags."),
        ("Twitter/X", "Create a concise post within 280 characters."),
        ("Email Newsletter", "Create a subject, greeting, informative body and CTA."),
    ]
    outputs = []
    for platform, instructions in platforms:
        result = one_agent_result(
            f"{platform} Content Specialist",
            f"Repurpose source content specifically for {platform}.",
            "You are a content marketing specialist. Preserve meaning and do not invent facts.",
            f"Source content:\n{data.input}\n\nPlatform: {platform}\n{instructions}",
        )
        outputs.append(f"### {platform}\n{result}")
    return {"output": "\n\n".join(outputs)}


# -------------------- Agent 9: Lead Qualification --------------------
@app.post("/agent9", tags=["Agent 9 - Lead Qualification"])
def agent9(data: Query):
    text = data.input
    # Try to extract a budget and timeline from the text; otherwise let the CrewAI
    # specialist qualify the lead based on the supplied information.
    budget_match = re.search(r"(?:budget|inr|rs\.?|₹)\s*[:=]?\s*([\d,]+)", text, re.I)
    timeline_match = re.search(r"(\d+)\s*(?:days|day)", text, re.I)
    budget = float(budget_match.group(1).replace(",", "")) if budget_match else 0
    timeline_days = int(timeline_match.group(1)) if timeline_match else 999
    if budget >= 50000 and timeline_days <= 30:
        category = "hot"
    elif budget >= 20000 or timeline_days <= 90:
        category = "warm"
    else:
        category = "cold"

    report = one_agent_result(
        "Lead Qualification Specialist",
        "Evaluate a lead and recommend suitable follow-up actions.",
        "You are an experienced sales specialist. Do not invent customer information.",
        f"Lead details:\n{text}\nInitial category: {category}\n"
        "Prepare overview, requirements, budget, timeline, qualification and next action. "
        "If information is missing, say so.",
    )
    return {"output": report}


# -------------------- Agent 10: Project Report and Approval --------------------
@app.post("/agent10", tags=["Agent 10 - Approval Report"])
def agent10(data: Query):
    report = one_agent_result(
        "Professional Report Writer",
        "Prepare a clear, accurate and professional project report.",
        "You are an experienced report writer. If information is missing, say so.",
        "Create a project report based on the following input. Include introduction, "
        "objectives, description, methodology, findings, conclusion and recommendations:\n"
        + data.input,
    )
    return {"output": report}


# Root URL redirects to Swagger docs, so opening the backend URL displays APIs.
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
