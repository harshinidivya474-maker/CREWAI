import os
import json
import io
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, HttpUrl
from crewai import Agent, Task, Crew, LLM, Process
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

app = FastAPI(
    title="CREWAI - 10 Agent Projects",
    description="FastAPI backend for ten CrewAI demonstrations.",
    version="1.0.0",
)

def get_llm():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    return LLM(model=os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite"),
               api_key=key, temperature=0.3)

def run_crew(agent_specs, task_specs, process=Process.sequential, manager=False):
    llm = get_llm()
    agents = []
    for spec in agent_specs:
        agents.append(Agent(
            role=spec["role"], goal=spec["goal"], backstory=spec["backstory"],
            llm=llm, verbose=False, allow_delegation=False, max_iter=4,
            tools=spec.get("tools", [])
        ))
    tasks = []
    for i, spec in enumerate(task_specs):
        tasks.append(Task(
            description=spec["description"],
            expected_output=spec["expected_output"],
            agent=agents[spec.get("agent", 0)],
            context=[tasks[j] for j in spec.get("context", [])],
            async_execution=spec.get("async_execution", False),
            output_pydantic=spec.get("output_pydantic")
        ))
    kwargs = dict(agents=agents, tasks=tasks, process=process, verbose=False)
    if manager:
        kwargs["manager_llm"] = llm
    result = Crew(**kwargs).kickoff()
    return str(result), result

def one_agent_result(role, goal, backstory, prompt):
    report, _ = run_crew(
        [{"role": role, "goal": goal, "backstory": backstory}],
        [{"description": prompt, "expected_output": "A clear, complete and useful response."}]
    )
    return report

# -------------------- Request models --------------------
class TopicRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=300)

class CompetitorRequest(BaseModel):
    urls: List[str] = Field(min_length=3, max_length=3)

class TranscriptRequest(BaseModel):
    transcript: str = Field(min_length=10, max_length=30000)

class CSVRequest(BaseModel):
    csv_data: str = Field(min_length=10, max_length=100000)

class SupportRequest(BaseModel):
    customer_name: str = "Customer"
    request: str = Field(min_length=3, max_length=5000)

class ResumeRequest(BaseModel):
    candidate_name: str = "Candidate"
    resume_text: str = Field(min_length=20, max_length=20000)
    job_role: str = Field(min_length=2, max_length=200)

class BugRequest(BaseModel):
    bug_title: str = Field(min_length=3, max_length=300)
    bug_description: str = Field(min_length=5, max_length=5000)

class ContentRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=300)
    original_content: str = Field(min_length=10, max_length=10000)

class LeadRequest(BaseModel):
    customer_name: str
    company_name: str = ""
    requirement: str
    budget: float = Field(ge=0)
    timeline_days: int = Field(ge=0)

class ApprovalRequest(BaseModel):
    project_title: str
    project_details: str = Field(min_length=10, max_length=15000)
    feedback: Optional[str] = None
    approved: Optional[bool] = None
    previous_draft: Optional[str] = None

# -------------------- Project 1: Sequential Blog Crew --------------------
@app.post("/agent1", tags=["Agent 1 - Blog Post Crew"])
def agent1(data: TopicRequest):
    llm = get_llm()
    researcher = Agent(role="Content Research Specialist",
        goal="Gather accurate information, examples and useful references.",
        backstory="You research topics carefully and distinguish facts from assumptions.",
        llm=llm, verbose=False, tools=[])
    writer = Agent(role="Professional Blog Creator",
        goal="Write an informative and engaging blog article.",
        backstory="You explain complex topics in simple reader-friendly language.",
        llm=llm, verbose=False)
    editor = Agent(role="Content Quality Editor",
        goal="Polish the article for clarity, grammar and factual consistency.",
        backstory="You are an experienced editor.",
        llm=llm, verbose=False)
    t1 = Task(description=f"Research this blog topic: {data.topic}. Provide key facts, examples and source URLs. Do not invent sources.",
              expected_output="Organized research notes with references.", agent=researcher)
    t2 = Task(description=f"Write an approximately 600-word blog post about {data.topic}, with title, introduction, headings and conclusion. Use the research notes.",
              expected_output="Complete Markdown blog article.", agent=writer, context=[t1])
    t3 = Task(description="Edit the article for grammar, clarity, flow and factual consistency. Preserve source links.",
              expected_output="Final polished Markdown blog article.", agent=editor, context=[t1, t2])
    result = Crew(agents=[researcher, writer, editor], tasks=[t1, t2, t3],
                  process=Process.sequential, verbose=False).kickoff()
    return {"project": "Blog Post Crew", "process": "sequential", "topic": data.topic, "result": str(result)}

# -------------------- Project 2: Hierarchical Competitor Analysis --------------------
def website_text(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=422, detail=f"Invalid URL: {url}")
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for el in soup(["script", "style", "nav", "footer", "noscript"]):
            el.decompose()
        return soup.get_text(" ", strip=True)[:7000]
    except requests.RequestException as exc:
        return f"Could not retrieve {url}: {exc}"

@app.post("/agent2", tags=["Agent 2 - Competitor Analysis"])
def agent2(data: CompetitorRequest):
    urls = [u.strip() for u in data.urls]
    if len(set(urls)) != 3:
        raise HTTPException(status_code=422, detail="Provide exactly three different competitor URLs.")
    gathered = {u: website_text(u) for u in urls}
    prompt = "Compare these three competitors using only the supplied website extracts.\n"
    for url, body in gathered.items():
        prompt += f"\nWEBSITE: {url}\nCONTENT: {body[:7000]}\n"
    prompt += """Create a competitor report with executive summary, profiles, products/services,
pricing (say Not found if absent), SWOT, opportunities, recommendations and source URLs.
Clearly label unknown information; do not invent prices or facts."""
    report, _ = run_crew(
        [{"role": "Competitor Analysis Manager", "goal": "Coordinate a fair, evidence-based competitor analysis.",
          "backstory": "You are a strategic market analyst who bases conclusions on supplied evidence."},
         {"role": "Market Research Analyst", "goal": "Extract comparable business facts.",
          "backstory": "You identify products, services, audiences and differentiators."},
         {"role": "Business Strategist", "goal": "Prepare practical recommendations.",
          "backstory": "You turn verified market findings into strategy."}],
        [{"description": "Analyze the supplied competitor website extracts and prepare structured comparison notes:\n" + prompt,
          "expected_output": "Evidence-based competitor research notes.", "agent": 1},
         {"description": "Using the competitor research, produce the final report with SWOT and recommendations.",
          "expected_output": "Complete competitor analysis report.", "agent": 2, "context": [0]}],
        process=Process.hierarchical, manager=True)
    return {"project": "Competitor Analysis Crew", "process": "hierarchical", "urls": urls, "report": report}

# -------------------- Project 3: Parallel Meeting Notes --------------------
@app.post("/agent3", tags=["Agent 3 - Meeting Notes"])
def agent3(data: TranscriptRequest):
    llm = get_llm()
    specs = [
        {"role": "Meeting Summarizer", "goal": "Summarize the meeting in five concise points.", "backstory": "You create accurate meeting summaries."},
        {"role": "Action Item Analyst", "goal": "Extract actions, owners and due dates.", "backstory": "You identify commitments and deadlines."},
        {"role": "Decision Logger", "goal": "Identify decisions made.", "backstory": "You maintain accurate decision records."},
        {"role": "Meeting Notes Organizer", "goal": "Combine outputs into polished notes.", "backstory": "You create clear professional meeting records."}
    ]
    agents = [Agent(role=s["role"], goal=s["goal"], backstory=s["backstory"], llm=llm, verbose=False) for s in specs]
    t1 = Task(description="Summarize this transcript in exactly five concise points:\n"+data.transcript,
              expected_output="Five-point meeting summary.", agent=agents[0], async_execution=True)
    t2 = Task(description="Extract action items, owners and due dates from this transcript. Use Not specified when absent:\n"+data.transcript,
              expected_output="Action list with owners and due dates.", agent=agents[1], async_execution=True)
    t3 = Task(description="List only decisions explicitly supported by this transcript:\n"+data.transcript,
              expected_output="List of meeting decisions.", agent=agents[2], async_execution=True)
    t4 = Task(description="Combine the previous outputs into Meeting Summary, Action Items and Decisions Made.",
              expected_output="Complete formatted meeting notes.", agent=agents[3], context=[t1,t2,t3])
    result = Crew(agents=agents, tasks=[t1,t2,t3,t4], process=Process.sequential, verbose=False).kickoff()
    return {"project": "Meeting Notes Crew", "execution": "parallel analysis tasks followed by formatting", "notes": str(result)}

# -------------------- Project 4: Structured Sales Report --------------------
class SalesAnalysis(BaseModel):
    total_rows: int
    total_sales: float
    average_sales: float
    top_products: List[str]
    key_findings: List[str]

@app.post("/agent4", tags=["Agent 4 - Structured Data Report"])
def agent4(data: CSVRequest):
    try:
        df = pd.read_csv(io.StringIO(data.csv_data))
        if "Sales" not in df.columns:
            if {"Quantity", "Price"}.issubset(df.columns):
                df["Sales"] = pd.to_numeric(df["Quantity"], errors="coerce") * pd.to_numeric(df["Price"], errors="coerce")
            else:
                raise HTTPException(status_code=422, detail="CSV must contain Sales or both Quantity and Price columns.")
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
        df = df.dropna(subset=["Sales"])
        if df.empty:
            raise HTTPException(status_code=422, detail="No valid sales records found.")
        top = []
        if "Product" in df.columns:
            top = df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(3).index.astype(str).tolist()
        total = float(df["Sales"].sum())
        stats = {"total_rows": int(len(df)), "total_sales": total,
                 "average_sales": float(df["Sales"].mean()), "top_products": top,
                 "key_findings": [f"Total revenue is {total:.2f}.",
                                  f"Average sales per record are {df['Sales'].mean():.2f}.",
                                  f"{len(df)} valid sales records were analyzed."]}
        structured = SalesAnalysis(**stats)
    except pd.errors.ParserError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {exc}")
    report = one_agent_result("Sales Report Reviewer", "Explain the supplied sales statistics clearly without changing figures.",
        "You are a careful business data reviewer.", "Prepare a concise report from these verified statistics:\n"+structured.model_dump_json(indent=2))
    return {"project": "Data Report Crew", "structured_output": structured.model_dump(), "report": report}

# -------------------- Project 5: Customer Support Router --------------------
@app.post("/agent5", tags=["Agent 5 - Customer Support Flow"])
def agent5(data: SupportRequest):
    text = data.request.lower()
    technical = ["error","bug","crash","login","password","not working","technical","website","app","network"]
    billing = ["payment","bill","billing","refund","invoice","charged","subscription","price","transaction","money"]
    category = "technical" if any(x in text for x in technical) else ("billing" if any(x in text for x in billing) else "general")
    report = one_agent_result(f"{category.title()} Support Specialist",
        "Give a polite, practical and accurate customer support response.",
        f"You handle {category} customer enquiries. Never invent company policies or promise unconfirmed actions.",
        f"Customer: {data.customer_name}\nRequest: {data.request}")
    return {"project": "Customer Support Flow", "category": category, "response": report}

# -------------------- Project 6: Resume Screening Router --------------------
@app.post("/agent6", tags=["Agent 6 - Resume Screening Flow"])
def agent6(data: ResumeRequest):
    role = data.job_role.lower()
    category = "python" if "python" in role else ("web" if any(x in role for x in ["web","frontend","front-end","react"]) else ("data" if any(x in role for x in ["data","analyst","analytics"]) else "general"))
    report = one_agent_result(f"{category.title()} Resume Screener",
        "Evaluate job-related qualifications and explain evidence, strengths and gaps.",
        "You are a fair recruiter. Do not make decisions based on personal characteristics; assess only job-related evidence.",
        f"Candidate: {data.candidate_name}\nJob role: {data.job_role}\nResume text:\n{data.resume_text}\nProvide relevant skills, strengths, gaps and a recommendation for further review.")
    return {"project": "Resume Screening Flow", "category": category, "candidate": data.candidate_name, "screening_report": report}

# -------------------- Project 7: Bug Triage Flow + Crews --------------------
@app.post("/agent7", tags=["Agent 7 - Bug Triage Flow"])
def agent7(data: BugRequest):
    text = (data.bug_title+" "+data.bug_description).lower()
    category = "frontend" if any(x in text for x in ["ui","css","button","layout","design","frontend","display","screen"]) else ("backend" if any(x in text for x in ["api","database","server","backend","authentication","request","sql"]) else "general")
    severity = "critical" if any(x in text for x in ["crash","data loss","security breach","system down","all users","critical"]) else ("high" if any(x in text for x in ["major","blocking","cannot access","not working","failure"]) else ("medium" if any(x in text for x in ["slow","minor","incorrect"]) else "low"))
    report = one_agent_result(f"{category.title()} Bug Analyst",
        "Prepare an evidence-aware bug triage report and debugging recommendations.",
        "You are a software debugging specialist. Do not claim to have reproduced or fixed the issue.",
        f"Bug title: {data.bug_title}\nDescription: {data.bug_description}\nInitial severity: {severity}\nExplain possible causes, affected component, debugging steps, possible solution and user impact. Separate facts from hypotheses.")
    return {"project": "Bug Triage Flow", "category": category, "severity": severity, "triage_report": report}

# -------------------- Project 8: Content Repurposing Flow + Crews --------------------
@app.post("/agent8", tags=["Agent 8 - Content Repurposing Flow"])
def agent8(data: ContentRequest):
    platforms = [
        ("LinkedIn", "Write a professional 120-180 word post with a strong opening and call to action."),
        ("Instagram", "Write an engaging caption with suitable emojis, call to action and 5-8 hashtags."),
        ("Twitter/X", "Write a concise post within 280 characters."),
        ("Email Newsletter", "Include a subject, greeting, informative body and call to action.")
    ]
    outputs = {}
    for platform, instructions in platforms:
        outputs[platform] = one_agent_result(f"{platform} Content Specialist",
            f"Repurpose source content specifically for {platform}.",
            "You are a social media and content marketing specialist. Preserve meaning and do not invent facts.",
            f"Topic: {data.topic}\nOriginal content:\n{data.original_content}\n{instructions}")
    return {"project": "Content Repurposing Flow", "content": outputs}

# -------------------- Project 9: Lead Qualification Router --------------------
@app.post("/agent9", tags=["Agent 9 - Lead Qualification Flow"])
def agent9(data: LeadRequest):
    if data.budget >= 50000 and data.timeline_days <= 30:
        category = "hot"
    elif data.budget >= 20000 or data.timeline_days <= 90:
        category = "warm"
    else:
        category = "cold"
    details = (f"Customer: {data.customer_name}\nCompany: {data.company_name}\n"
               f"Requirement: {data.requirement}\nBudget INR: {data.budget}\n"
               f"Timeline: {data.timeline_days} days\nCategory: {category}")
    report = one_agent_result("Lead Qualification Specialist",
        "Evaluate the lead and recommend suitable follow-up actions.",
        "You are an experienced sales specialist. Do not invent customer information.",
        details+"\nPrepare overview, requirement, budget analysis, timeline, category, interest, sales strategy and next action.")
    return {"project": "Lead Qualification Flow", "category": category, "qualification_report": report}

# -------------------- Project 10: Approval Report + Human Feedback --------------------
@app.post("/agent10", tags=["Agent 10 - Approval Report Flow"])
def agent10(data: ApprovalRequest):
    if data.previous_draft:
        if not data.feedback or not data.feedback.strip():
            raise HTTPException(status_code=422, detail="Provide feedback when submitting a previous_draft for revision.")
        report = one_agent_result("Report Revision Specialist",
            "Revise a report according to human feedback while preserving supported facts.",
            "You are an expert editor. Do not invent facts.",
            f"Project: {data.project_title}\nOriginal report:\n{data.previous_draft}\nHuman feedback:\n{data.feedback}\nReturn the complete revised report.")
        status = "APPROVED" if data.approved is True else "PENDING HUMAN APPROVAL"
        return {"project": "Approval Report Flow", "status": status, "feedback": data.feedback, "report": report}
    draft = one_agent_result("Professional Report Writer",
        "Prepare a clear, accurate and professional project report.",
        "You are an experienced report writer. If information is missing, say so rather than inventing it.",
        f"Project title: {data.project_title}\nProject details: {data.project_details}\nInclude introduction, objectives, description, methodology, findings, conclusion and recommendations.")
    if data.approved is True:
        status = "APPROVED"
    else:
        status = "PENDING HUMAN APPROVAL"
    return {"project": "Approval Report Flow", "status": status, "draft_report": draft,
            "next_step": "Review the draft. To request changes, call /agent10 again with previous_draft and feedback. To mark it approved, submit approved=true."}

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
