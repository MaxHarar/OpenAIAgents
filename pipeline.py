
import asyncio
import csv
import json
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Literal, TypeVar

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agents import (
    Agent,
    ModelBehaviorError,
    ModelResponse,
    ModelSettings,
    RunHooks,
    Runner,
    WebSearchTool,
)

load_dotenv()

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

MODEL = "gpt-4.1"
MAX_CONCURRENCY = 5  # parallel companies in flight
MAX_RETRIES = 3
DB_PATH = Path("crm.db")

# Standard tier for gpt-4.1, USD per 1M tokens (developers.openai.com/api/docs/pricing).
# cost_usd includes web search call fees but not cached-input discounts, so it can run slightly high.
PRICE_PER_M_INPUT = 2.00
PRICE_PER_M_OUTPUT = 8.00
PRICE_PER_1K_WEB_SEARCH = 10.00  # WebSearchTool sends type "web_search"; content tokens bill as input

# Only retry errors that a retry can fix. A 404 or bad key should fail fast.
RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    ModelBehaviorError,  # model returned output that failed the schema
)

# ----------------------------------------------------------------------------
# Schemas (Pydantic = structured outputs + validation)
# ----------------------------------------------------------------------------


class Contact(BaseModel):
    name: str
    title: str


class Company(BaseModel):
    name: str
    domain: str = Field(description="Bare primary domain, e.g. gxo.com. No https or www.")
    industry: str
    hq_country: str
    employee_estimate: int | None = Field(description="Best headcount estimate, null if unknown")
    physical_operations: list[str] = Field(
        description="Kinds of physical sites: warehouses, factories, farms, stores, etc."
    )
    automation_signals: list[str] = Field(
        description="Concrete evidence of robotics/automation investment or labor shortages"
    )
    key_contacts: list[Contact] = Field(
        description="Up to 3 likely buyers in operations, supply chain, or automation leadership"
    )
    sources: list[str] = Field(description="URLs relied on")


class ICPScore(BaseModel):
    score: int = Field(description="0 to 100")
    tier: Literal["A", "B", "C", "disqualified"]
    qualified: bool
    criteria_hits: list[str]
    criteria_misses: list[str]
    rationale: str = Field(description="2 to 3 sentences citing specific evidence")


class CRMRecord(BaseModel):
    company: Company
    score: ICPScore
    stage: Literal["new", "qualified", "disqualified"]
    run_id: str


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

ENRICH_PROMPT = """
# Role
You research one company for a robotics sales team.

# Task
Use web search to fill every field of the output schema for the company in the input.

# Rules
- Use only facts you found. Unknown fields get an empty list or null.
- domain is the bare primary domain, e.g. gxo.com.
- automation_signals: concrete evidence only (a robot pilot, an automation program,
  a public statement about labor shortages). One short line each.
- key_contacts: at most 3 current leaders in operations, supply chain, manufacturing,
  or automation. Empty list if not found.
- sources: the URLs you relied on.
"""

ICP_PROMPT = """
# Role
You qualify accounts for a robotics company that builds general-purpose robots
for homes, workplaces, factories, and farms.

# Ideal customer profile
1. Large physical operations: warehouses, factories, distribution centers, or farms.
2. Repetitive manual work at scale: picking, tote moving, kitting, machine tending.
3. Automation appetite: robotics pilots, automation budget, or stated labor shortages.
4. Scale: roughly 1,000+ employees or many sites, so a pilot can expand.
5. US operations, since early deployments are easier near the team.

# Scoring
- score: 0 to 100. Each criterion fully met is worth about 20 points.
- tier: A (80+), B (60 to 79), C (40 to 59), disqualified (below 40).
- qualified: true only for tier A or B.
- criteria_hits / criteria_misses: criterion number plus evidence,
  e.g. "3: Digit humanoid pilot at a GXO facility".
- rationale: 2 to 3 sentences grounded only in the input. Never invent evidence.
  Missing data counts as a miss.
"""

enricher = Agent(
    name="Enricher",
    instructions=ENRICH_PROMPT,
    model=MODEL,
    tools=[WebSearchTool()],
    output_type=Company,
    model_settings=ModelSettings(temperature=0.2),
)

scorer = Agent(
    name="ICP Scorer",
    instructions=ICP_PROMPT,
    model=MODEL,
    output_type=ICPScore,
    model_settings=ModelSettings(temperature=0),  # scoring should be repeatable
)

# ----------------------------------------------------------------------------
# Run stats: tokens, cost, failures
# ----------------------------------------------------------------------------


@dataclass
class RunStats:
    input_tokens: int = 0
    output_tokens: int = 0
    llm_requests: int = 0
    web_search_calls: int = 0
    retries: int = 0
    failures: list[str] = field(default_factory=list)

    def add_response(self, response: ModelResponse) -> None:
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.llm_requests += response.usage.requests
        self.web_search_calls += count_web_searches(response.output)

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1e6 * PRICE_PER_M_INPUT
            + self.output_tokens / 1e6 * PRICE_PER_M_OUTPUT
            + self.web_search_calls / 1000 * PRICE_PER_1K_WEB_SEARCH
        )


def count_web_searches(output) -> int:
    """Usage has no search counter, so read the response output. OpenAI bills search actions."""
    return sum(
        1
        for item in output
        if getattr(item, "type", None) == "web_search_call" and item.action.type == "search"
    )


class StatsHooks(RunHooks):
    """Record each response as it arrives, so attempts that later fail are still counted."""

    def __init__(self, stats: RunStats) -> None:
        self.stats = stats

    async def on_llm_end(self, context, agent, response: ModelResponse) -> None:
        self.stats.add_response(response)


# ----------------------------------------------------------------------------
# Retries + concurrency
# ----------------------------------------------------------------------------

T = TypeVar("T")


async def run_with_retry(agent: Agent, prompt: str, stats: RunStats, label: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await Runner.run(agent, prompt, hooks=StatsHooks(stats))
            return result.final_output
        except RETRYABLE as e:
            if attempt == MAX_RETRIES:
                raise
            stats.retries += 1
            wait = 2**attempt + random.random()  # exponential backoff + jitter
            print(f"  retry {attempt} [{agent.name}] {label}: {type(e).__name__}, waiting {wait:.1f}s")
            await asyncio.sleep(wait)


async def guarded(
    sem: asyncio.Semaphore, stats: RunStats, label: str, fn: Callable[[], Awaitable[T]]
) -> T | None:
    """Cap concurrency and turn one company's failure into a logged skip."""
    async with sem:
        try:
            return await fn()
        except Exception as e:
            stats.failures.append(f"{label}: {type(e).__name__}: {e}")
            print(f"  FAILED {label}: {type(e).__name__}")
            return None


# ----------------------------------------------------------------------------
# Dedupe (entity resolution, simple version)
# ----------------------------------------------------------------------------

SUFFIXES = r"\b(inc|incorporated|corp|corporation|co|company|llc|ltd|plc|gmbh|ag|sa|group|holdings)\b"


def ascii_dashes(s: str) -> str:
    """Models sometimes emit Unicode dashes (e.g. U+2011), which break exact-match keys."""
    return re.sub(r"[‐-―−﹘﹣－]", "-", s)


def norm_name(name: str) -> str:
    n = ascii_dashes(name.lower())
    n = re.sub(r"[.,&()]", " ", n)
    n = re.sub(SUFFIXES, " ", n)
    return re.sub(r"\s+", " ", n).strip()


def norm_domain(domain: str) -> str:
    d = ascii_dashes(domain.lower().strip())
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    return d.split("/")[0]


def dedupe_inputs(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in rows:
        keys = {"name:" + norm_name(r["name"])}
        if r.get("domain"):
            keys.add("domain:" + norm_domain(r["domain"]))
        if keys & seen:
            print(f"  dup input skipped: {r['name']}")
            continue
        seen |= keys
        out.append(r)
    return out


def dedupe_enriched(companies: list[Company]) -> list[Company]:
    seen: set[str] = set()
    out = []
    for c in companies:
        key = norm_domain(c.domain)
        if key in seen:
            print(f"  dup after enrichment skipped: {c.name} ({key})")
            continue
        seen.add(key)
        out.append(c)
    return out


# ----------------------------------------------------------------------------
# Deterministic guardrails on LLM output
# ----------------------------------------------------------------------------


def tier_for(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "disqualified"


def enforce_score_rules(s: ICPScore) -> ICPScore:
    """The model proposes; code enforces. Keeps tier/qualified consistent with score."""
    score = max(0, min(100, s.score))
    tier = tier_for(score)
    return s.model_copy(update={"score": score, "tier": tier, "qualified": tier in ("A", "B")})


# ----------------------------------------------------------------------------
# SQLite CRM
# ----------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY,
    domain TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    industry TEXT,
    hq_country TEXT,
    employee_estimate INTEGER,
    physical_operations TEXT,
    automation_signals TEXT,
    sources TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    title TEXT,
    UNIQUE (account_id, name)
);
CREATE TABLE IF NOT EXISTS deals (
    id INTEGER PRIMARY KEY,
    account_id INTEGER UNIQUE NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (stage IN ('new', 'qualified', 'disqualified')),
    icp_score INTEGER,
    tier TEXT,
    rationale TEXT,
    criteria_hits TEXT,
    criteria_misses TEXT,
    run_id TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT,
    finished_at TEXT,
    companies_in INTEGER,
    companies_written INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    llm_requests INTEGER,
    retries INTEGER,
    cost_usd REAL,
    failures TEXT,
    web_search_calls INTEGER
);
"""


def migrate(conn: sqlite3.Connection) -> None:
    """Add columns to DBs created before they existed. Old runs keep NULL: the count is unknown."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    if "web_search_calls" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN web_search_calls INTEGER")


def write_to_crm(conn: sqlite3.Connection, rec: CRMRecord) -> None:
    """Upsert so reruns update rows instead of duplicating them."""
    now = datetime.now(timezone.utc).isoformat()
    c, s = rec.company, rec.score
    domain = norm_domain(c.domain)

    conn.execute(
        """
        INSERT INTO accounts (domain, name, industry, hq_country, employee_estimate,
                              physical_operations, automation_signals, sources, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(domain) DO UPDATE SET
            name=excluded.name, industry=excluded.industry, hq_country=excluded.hq_country,
            employee_estimate=excluded.employee_estimate,
            physical_operations=excluded.physical_operations,
            automation_signals=excluded.automation_signals,
            sources=excluded.sources, updated_at=excluded.updated_at
        """,
        (
            domain, c.name, c.industry, c.hq_country, c.employee_estimate,
            json.dumps(c.physical_operations), json.dumps(c.automation_signals),
            json.dumps(c.sources), now,
        ),
    )
    account_id = conn.execute("SELECT id FROM accounts WHERE domain = ?", (domain,)).fetchone()[0]

    for person in c.key_contacts[:3]:
        conn.execute(
            """
            INSERT INTO contacts (account_id, name, title) VALUES (?, ?, ?)
            ON CONFLICT(account_id, name) DO UPDATE SET title=excluded.title
            """,
            (account_id, person.name, person.title),
        )

    conn.execute(
        """
        INSERT INTO deals (account_id, stage, icp_score, tier, rationale,
                           criteria_hits, criteria_misses, run_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id) DO UPDATE SET
            stage=excluded.stage, icp_score=excluded.icp_score, tier=excluded.tier,
            rationale=excluded.rationale, criteria_hits=excluded.criteria_hits,
            criteria_misses=excluded.criteria_misses, run_id=excluded.run_id,
            updated_at=excluded.updated_at
        """,
        (
            account_id, rec.stage, s.score, s.tier, s.rationale,
            json.dumps(s.criteria_hits), json.dumps(s.criteria_misses), rec.run_id, now,
        ),
    )


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------


async def enrich(row: dict, stats: RunStats) -> Company:
    prompt = f"Company: {row['name']}\nKnown domain: {row.get('domain') or 'unknown'}"
    return await run_with_retry(enricher, prompt, stats, row["name"])


async def score(company: Company, stats: RunStats) -> ICPScore:
    raw = await run_with_retry(scorer, company.model_dump_json(indent=2), stats, company.name)
    return enforce_score_rules(raw)


async def main(csv_path: str) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    stats = RunStats()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("name", "").strip()]
    inputs = dedupe_inputs(rows)
    print(f"{len(rows)} rows in, {len(inputs)} after input dedupe")

    # Phase 1: enrich in parallel
    enriched = await asyncio.gather(
        *(guarded(sem, stats, r["name"], lambda r=r: enrich(r, stats)) for r in inputs)
    )
    companies = dedupe_enriched([c for c in enriched if c is not None])
    print(f"{len(companies)} companies enriched and deduped")

    # Phase 2: score in parallel (dedupe first so we never pay to score a duplicate)
    scores = await asyncio.gather(
        *(guarded(sem, stats, c.name, lambda c=c: score(c, stats)) for c in companies)
    )

    # Phase 3: write (plain code, no LLM)
    records = [
        CRMRecord(
            company=c,
            score=s,
            stage="qualified" if s.qualified else "disqualified",
            run_id=run_id,
        )
        for c, s in zip(companies, scores)
        if s is not None
    ]

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    migrate(conn)
    with conn:  # one transaction
        for rec in records:
            write_to_crm(conn, rec)
        conn.execute(
            """
            INSERT INTO runs (run_id, started_at, finished_at, companies_in, companies_written,
                              input_tokens, output_tokens, llm_requests, retries, cost_usd,
                              failures, web_search_calls)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, started, datetime.now(timezone.utc).isoformat(),
                len(rows), len(records), stats.input_tokens, stats.output_tokens,
                stats.llm_requests, stats.retries, stats.cost_usd, json.dumps(stats.failures),
                stats.web_search_calls,
            ),
        )
    conn.close()

    # Summary
    elapsed = time.perf_counter() - t0
    print("\n" + f"{'Company':<32}{'Score':>6}  Tier")
    for rec in sorted(records, key=lambda r: r.score.score, reverse=True):
        print(f"{rec.company.name[:31]:<32}{rec.score.score:>6}  {rec.score.tier}")
    print(
        f"\nrun {run_id}: {len(records)} written, {len(stats.failures)} failed, "
        f"{stats.retries} retries, {stats.llm_requests} LLM requests, "
        f"{stats.web_search_calls} web searches, "
        f"{stats.input_tokens:,} in / {stats.output_tokens:,} out tokens, "
        f"${stats.cost_usd:.4f}, {elapsed:.1f}s"
    )
    for fail in stats.failures:
        print(f"  failure: {fail}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python pipeline.py companies.csv")
    asyncio.run(main(sys.argv[1]))
