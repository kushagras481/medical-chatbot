#!/usr/bin/env python3
# rag_medical_bot_enhanced.py – PubMed‑augmented Gemini chatbot with self‑feedback RAG
# ---------------------------------------------------------------------------------
#   • One‑shot PubMed query generation via Gemini
#   • Iterative self‑feedback (default 2 rounds) to improve citing coverage
#   • Citation verification via Gemini Yes/No entailment prompt
#   • Automatic answer regeneration when invalid citations are detected (option B)
#   • CER warning if < 0.50 → user alert in Korean
#   • Diagnostic JSON log written to citations.json
#   • Lightweight – no local NLI model required
#   • Returns: {'answer', 'references', 'query', 'cer', 'citations', 'warning'}
#   • DEBUG=1 → detailed prompt / response / PubMed logs
# ---------------------------------------------------------------------------------

from __future__ import annotations
import os
import re
import json
import textwrap
from pathlib import Path
from functools import lru_cache
from typing import List, Dict
from datetime import datetime

import google.generativeai as genai
from Bio import Entrez
from dotenv import load_dotenv

# ───────────────────────────────
# 0) Environment + Globals
# ───────────────────────────────
load_dotenv()
DEBUG_RAG = os.getenv("DEBUG", "0") == "1"    # export DEBUG=1 to enable

CACHE_DIR = Path(os.getenv("RAG_CACHE", ".cache"))
CACHE_DIR.mkdir(exist_ok=True)
LOG_PATH = CACHE_DIR / "citations.json"          # diagnostic log


def _dbg(msg: str, block: bool = False):
    """Debug print helper respecting DEBUG env."""
    if DEBUG_RAG:
        if block:
            print("\n" + "─" * 70)
        print(msg)


GEM_KEY = os.getenv("GEMINI_API_KEY")
if not GEM_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

# Shared Gemini model instance (flash for speed)
genai.configure(api_key=GEM_KEY)
_GEMINI = genai.GenerativeModel("gemini-2.0-flash")

# Entrez (NCBI) settings
Entrez.email = os.getenv("ENTREZ_EMAIL", "anon@example.com")

# ───────────────────────────────
# 1) Prompt templates
# ───────────────────────────────
PROMPT_QUERY = textwrap.dedent("""\
You are an expert at translating user medical questions into effective PubMed search queries.
Convert the following user question into a *single* concise PubMed search query that combines
MeSH terms and free‑text TIAB keywords. Wrap the query in double‑quotes and tag fields using
[MeSH Terms] or [TIAB]. Do *not* add extra commentary.

USER_Q: {question}
""")

PROMPT_ANSWER = textwrap.dedent("""\
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
""")

SELF_FEEDBACK_PROMPT = textwrap.dedent("""\
You are a critical reviewer. Read the draft answer and the numbered abstracts below.
Identify statements that need stronger evidence or lack citations, and list up to three
specific medical topics or keywords that should be searched for additional references.
Respond in the format:
  search for <topic 1>
  search for <topic 2>
  ...
If no additional searches seem necessary, reply with "no additional search".

Draft answer:
{draft}

Abstracts:
{numbered}
""")

ENTAIL_PROMPT = textwrap.dedent("""\
You are a medical expert. Determine if the following sentence is *clearly supported*
by the abstract below. Respond only with one word: Yes or No.

Sentence:
"{sentence}"

Abstract:
{abstract}
""")

# ───────────────────────────────
# 2) Gemini helper with debug
# ───────────────────────────────

def _call_gemini(prompt: str) -> str:
    """Wrapper for Gemini calls with optional debug prints."""
    _dbg("LLM PROMPT ↓", block=True)
    _dbg(prompt[:1200] + (" …[truncated]…" if len(prompt) > 1200 else ""))
    resp = _GEMINI.generate_content(prompt).text.strip()
    _dbg("LLM RESPONSE ↓", block=True)
    _dbg(resp)
    return resp


# ───────────────────────────────
# 3) Utility
# ───────────────────────────────

def _redact(txt: str) -> str:
    """Simple PII redaction for obvious tokens."""
    for term in ["SSN", "social security", "address", "phone", "email"]:
        txt = re.sub(re.escape(term), "[REDACTED]", txt, flags=re.I)
    return txt


# ───────────────────────────────
# 4) PubMed helpers (cached)
# ───────────────────────────────

@lru_cache(maxsize=256)
def _esearch(query: str, k: int = 20) -> List[str]:
    _dbg(f"📡 ESearch → {query}")
    try:
        ids = Entrez.read(
            Entrez.esearch(db="pubmed", term=query, retmax=k, sort="relevance")
        )["IdList"]
        _dbg(f"✅ {len(ids)} IDs")
        return ids
    except Exception as e:
        _dbg(f"❌ ESearch error: {e}")
        return []


@lru_cache(maxsize=256)
def _efetch(ids_csv: str) -> List[Dict]:
    _dbg(f"📥 EFetch → {ids_csv[:80]}…")
    try:
        recs = Entrez.read(
            Entrez.efetch(db="pubmed", id=ids_csv, rettype="xml", retmode="xml")
        )
    except Exception as e:
        _dbg(f"❌ EFetch error: {e}")
        return []

    arts: List[Dict] = []
    for art in recs.get("PubmedArticle", []):
        med = art["MedlineCitation"]
        pmid = str(med["PMID"])
        art_info = med.get("Article", {})
        title = art_info.get("ArticleTitle", "")
        authors = [
            f"{a.get('LastName')} {a.get('Initials')}"
            for a in art_info.get("AuthorList", [])
            if a.get("LastName")
        ]
        abst = art_info.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(x) for x in (abst if isinstance(abst, list) else [abst]))
        arts.append({"pmid": pmid, "title": title, "authors": authors, "abstract": abstract})

    _dbg(f"✅ abstracts fetched: {len(arts)}")
    return arts


def _fetch_articles(query: str, limit: int = 10) -> List[Dict]:
    return _efetch(",".join(_esearch(query, limit))) if query else []


# ───────────────────────────────
# 5) Generation & feedback helpers
# ───────────────────────────────

def _generate_answer(pub_query: str, arts: List[Dict]) -> str:
    numbered = "\n".join(f"[{i}] {a['abstract']}" for i, a in enumerate(arts, 1))
    return _call_gemini(
        PROMPT_ANSWER.format(query=pub_query, numbered=numbered or "(none)")
    )


def _self_feedback(draft: str, arts: List[Dict]) -> str:
    numbered = "\n".join(f"[{i}] {a['abstract']}" for i, a in enumerate(arts, 1))
    return _call_gemini(
        SELF_FEEDBACK_PROMPT.format(draft=draft, numbered=numbered)
    )


def _extract_topics(feedback: str) -> List[str]:
    return (
        []
        if "no additional search" in feedback.lower()
        else re.findall(r"search for ([A-Za-z0-9 \-]+)", feedback, flags=re.I)
    )


# ───────────────────────────────
# 6) Citation verification via Gemini
# ───────────────────────────────
def _verify_citations_llm(answer: str, arts: List[Dict]) -> List[Dict]:
    results: List[Dict] = []
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    for sent in sentences:
        for tag in re.findall(r"\[(\d+)\]", sent):
            idx = int(tag) - 1
            if 0 <= idx < len(arts):
                verdict = _call_gemini(
                    ENTAIL_PROMPT.format(
                        sentence=sent, abstract=arts[idx]["abstract"][:1500]
                    )
                ).lower()
                valid = verdict.startswith("yes")
            else:
                verdict = "unverifiable"
                valid = False

            results.append({
                "sentence": sent,
                "citation": idx + 1,
                "verdict": verdict,
                "valid": valid,
            })
    return results



def _cer(results: List[Dict]) -> float | None:
    valid_results = [r for r in results if r["verdict"] != "unverifiable"]
    return None if not valid_results else sum(r["valid"] for r in valid_results) / len(valid_results)



# ───────────────────────────────
# 7) Diagnostic log helper
# ───────────────────────────────

def _write_log(payload: Dict[str, object]) -> None:
    """Append diagnostic info to citations.json inside CACHE_DIR."""
    try:
        if LOG_PATH.exists():
            data = json.loads(LOG_PATH.read_text())
            if isinstance(data, list):
                logs: List[Dict] = data
            else:
                logs = [data]  # legacy single‑object file
        else:
            logs = []
        logs.append(payload)
        LOG_PATH.write_text(json.dumps(logs, ensure_ascii=False, indent=2))
        _dbg(f"📝 Log written to {LOG_PATH}")
    except Exception as e:
        _dbg(f"❌ Failed to write log: {e}")


# ───────────────────────────────
# 8) Main RAG function
# ───────────────────────────────

def answer_medical_question(
    question: str,
    *,
    max_abstracts: int = 10,
    iterations: int = 2,
    max_regens: int = 1,  # new: max full answer regenerations when invalid citations detected
) -> Dict[str, object]:
    """Return answer + metadata for a user question with RAG pipeline."""

    # 1) redact → PubMed query
    pub_query = _call_gemini(PROMPT_QUERY.format(question=_redact(question))).strip("\"")
    _dbg(f"🔍 PubMed query → {pub_query}", block=True)

    # 2) fetch & initial answer
    arts: List[Dict] = _fetch_articles(pub_query, max_abstracts)
    _dbg(f"📑 fetched {len(arts)} abstracts")
    draft = _generate_answer(pub_query, arts)

    # 3) self‑feedback loop (to broaden coverage)
    for it in range(max(0, iterations - 1)):
        fb = _self_feedback(draft, arts)
        tops = _extract_topics(fb)
        _dbg(f"🔁 iter {it + 1} topics → {tops or 'none'}")
        if tops:
            seen = {a["pmid"] for a in arts}
            for tp in tops:
                for art in _fetch_articles(f"{tp}[MeSH Terms]", 5):
                    if art["pmid"] not in seen:
                        seen.add(art["pmid"])
                        arts.append(art)
            _dbg(f"➕ total abstracts now {len(arts)}")
        draft = _generate_answer(pub_query, arts)

    _dbg("✅ draft prepared; starting citation verification", block=True)

    # 4) citation verification & optional regeneration
    regen_left = max_regens
    cites = _verify_citations_llm(draft, arts)
    cer = _cer(cites)

    while regen_left > 0 and cites and any(not c["valid"] for c in cites):
        regen_left -= 1
        _dbg("♻️ Invalid citations detected – regenerating answer", block=True)
        draft = _generate_answer(pub_query, arts)  # full regeneration (Option B)
        cites = _verify_citations_llm(draft, arts)
        cer = _cer(cites)

    _dbg(
        f"📊 Citation Entailment Rate (CER): {cer:.2%}" if cer is not None else "📊 No citations",
        block=True,
    )

    # 5) Build reference list
    refs = (
        [f"[{i}] {', '.join(a['authors'])}, {a['title']} (PMID:{a['pmid']})" for i, a in enumerate(arts, 1)]
        if arts
        else []
    )

    # 6) Warning flag if CER < 0.50
    warning_msg = (
        "Warning: Several Citations might not be appropriate." if (cer is not None and cer < 0.5) else None
    )

    # 7) Diagnostic log
    _write_log(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "query": pub_query,
            "cer": cer,
            "cited_sentences": cites,
        }
    )
    if warning_msg:
        print("Final Answer: "+ draft + warning_msg)
    else:
        print("Final Answer: "+ draft)

    return {
        "answer": draft,
        "references": refs,
        "query": pub_query,
        "cer": cer,
        "citations": cites,
        "warning": warning_msg,
    }

