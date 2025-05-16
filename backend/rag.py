# rag_medical_bot.py – PubMed‑augmented Gemini chatbot with self‑feedback RAG
# -------------------------------------------------------------------------
# For web‑service use: low latency, no heavy NLI verification, but *keeps*
# a lightweight self‑feedback loop to refine answers.
#
# Key characteristics
#   • One‑shot PubMed query generation via Gemini
#   • Iterative self‑feedback (default 2 rounds) to improve citing coverage
#   • No transformer‑based NLI checks → lighter dependencies & faster
#   • Returns separate fields: answer, references (list), query
#   • PII redaction for basic privacy
#
# Environment variables (.env):
#   GEMINI_API_KEY – Google Generative AI key
#   ENTREZ_EMAIL    – contact email for PubMed API etiquette
# -------------------------------------------------------------------------
from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Set

import google.generativeai as genai
from Bio import Entrez
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ───────────────────────────────
# 0) Environment + Globals
# ───────────────────────────────
GEM_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEM_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

genai.configure(api_key=GEM_KEY)
_GEMINI = genai.GenerativeModel("gemini-2.0-flash")

Entrez.email = os.getenv("ENTREZ_EMAIL", "anon@example.com")
CACHE_DIR = Path(os.getenv("RAG_CACHE", ".cache"))
CACHE_DIR.mkdir(exist_ok=True)

# ───────────────────────────────
# 1) Prompt templates
# ───────────────────────────────
PROMPT_QUERY = textwrap.dedent(
    """
    You are an expert at translating user medical questions into effective PubMed search queries.
    Convert the following user question into a *single* concise PubMed search query that combines
    MeSH terms and free‑text TIAB keywords. Wrap the query in double‑quotes and tag fields using
    [MeSH Terms] or [TIAB]. Do *not* add extra commentary.

    USER_Q: {question}
    """
)

PROMPT_ANSWER = textwrap.dedent(
    """
    You are a helpful medical assistant.

    PubMed search query:
    {query}

    Relevant abstracts (numbered 1–N):
    {numbered}

    Write a concise answer (≤2 short paragraphs) in plain language suitable for
    a middle‑school reader. Include inline citations like [1], [2] *only* when the
    statement is directly supported by the corresponding abstract. Finish with:
    "Consult a healthcare professional for personalized advice." If no abstracts were
    retrieved, start with: "No articles retrieved; answer is based on general medical knowledge.".
    """
)

SELF_FEEDBACK_PROMPT = textwrap.dedent(
    """
    You are a critical reviewer. Read the draft answer and the numbered abstracts below.
    Identify statements that need stronger evidence or lack citations, and list up to three
    specific medical topics or keywords that should be searched for additional references.
    Respond in the format:
      search for <topic 1>\n
      search for <topic 2>\n
      ...
    If no additional searches seem necessary, reply with "no additional search".

    Draft answer:
    {draft}

    Abstracts:
    {numbered}
    """
)

# ───────────────────────────────
# 2) Helpers
# ───────────────────────────────

def _call_gemini(prompt: str) -> str:
    return _GEMINI.generate_content(prompt).text.strip()


def _redact(text: str) -> str:
    sensitive = ["SSN", "social security", "address", "phone", "email"]
    out = text
    for term in sensitive:
        out = re.sub(re.escape(term), "[REDACTED]", out, flags=re.I)
    return out


# ───────────────────────────────
# 3) PubMed utilities with caching
# ───────────────────────────────
@lru_cache(maxsize=256)
def _esearch(query: str, max_results: int = 20) -> List[str]:
    try:
        h = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        return Entrez.read(h).get("IdList", [])
    except Exception:
        return []


@lru_cache(maxsize=256)
def _efetch(ids_csv: str) -> List[Dict]:
    try:
        h = Entrez.efetch(db="pubmed", id=ids_csv, rettype="xml", retmode="xml")
        recs = Entrez.read(h)
    except Exception:
        return []

    arts = []
    for art in recs.get("PubmedArticle", []):
        med = art["MedlineCitation"]
        pmid = str(med.get("PMID"))
        info = med.get("Article", {})
        title = info.get("ArticleTitle", "")
        authors = [f"{a.get('LastName')} {a.get('Initials')}" for a in info.get("AuthorList", []) if a.get("LastName")]
        abst = info.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(x) for x in (abst if isinstance(abst, list) else [abst]))
        arts.append({"pmid": pmid, "title": title, "authors": authors, "abstract": abstract})
    return arts


def _fetch_articles(query: str, limit: int = 10) -> List[Dict]:
    ids = _esearch(query, limit)
    return _efetch(",".join(ids)) if ids else []


# ───────────────────────────────
# 4) Self‑feedback helpers
# ───────────────────────────────

def _generate_answer(pub_query: str, articles: List[Dict]) -> str:
    numbered = "\n".join(f"[{i}] {a['abstract']}" for i, a in enumerate(articles, 1))
    return _call_gemini(PROMPT_ANSWER.format(query=pub_query, numbered=numbered or "(none)"))


def _self_feedback(draft: str, articles: List[Dict]) -> str:
    numbered = "\n".join(f"[{i}] {a['abstract']}" for i, a in enumerate(articles, 1))
    return _call_gemini(SELF_FEEDBACK_PROMPT.format(draft=draft, numbered=numbered))


def _extract_topics(feedback: str) -> List[str]:
    if "no additional search" in feedback.lower():
        return []
    return re.findall(r"search for ([A-Za-z0-9 \-]+)", feedback, flags=re.I)


# ───────────────────────────────
# 5) Main RAG function (with iterations)
# ───────────────────────────────

def answer_medical_question(question: str, *, max_abstracts: int = 10, iterations: int = 2) -> Dict[str, object]:
    """Return dict with keys: answer, references (list[str]), query."""

    # 1) redact PII & craft PubMed query
    pub_query = _call_gemini(PROMPT_QUERY.format(question=_redact(question))).strip().strip('"')

    # 2) initial article fetch & answer
    articles: List[Dict] = _fetch_articles(pub_query, max_abstracts)
    draft = _generate_answer(pub_query, articles)

    # 3) iterative self‑feedback loop (no heavy NLI)
    for _ in range(max(0, iterations - 1)):
        feedback = _self_feedback(draft, articles)
        topics = _extract_topics(feedback)
        if topics:
            seen: Set[str] = {a['pmid'] for a in articles}
            for tp in topics:
                new = _fetch_articles(f"{tp}[MeSH Terms]", 5)
                for art in new:
                    if art['pmid'] not in seen:
                        seen.add(art['pmid'])
                        articles.append(art)
        # regenerate answer with (possibly) expanded article set
        draft = _generate_answer(pub_query, articles)

    # 4) build reference list
    references = [f"[{i}] {', '.join(a['authors'])}, {a['title']} (PMID: {a['pmid']})" for i, a in enumerate(articles, 1)] if articles else []

    return {
        "answer": draft,
        "references": references,
        "query": pub_query,
    }


# ───────────────────────────────
# 6) Simple CLI for quick tests
# ───────────────────────────────
if __name__ == "__main__":
    try:
        while True:
            q = input("\n💬 Ask a medical question (or 'q' to quit): ").strip()
            if q.lower() in {"q", "quit", "exit"}: break
            resp = answer_medical_question(q, iterations=3)
            print("\n=== PubMed query ===\n", resp['query'])
            print("\n=== Answer ===\n", resp['answer'])
            if resp['references']:
                print("\n=== References ===")
                for ref in resp['references']:
                    print(ref)
    except KeyboardInterrupt:
        print("\nBye")
