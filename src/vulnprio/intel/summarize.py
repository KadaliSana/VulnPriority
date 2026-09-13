"""The model-written summary, and the rules that decide whether one may exist at all.

The framework's report is otherwise generated deterministically from data, and DESIGN 6.2
says so in as many words. This is the one exception, and it is allowed only under
conditions that keep the exception honest:

* It is labelled. :class:`~vulnprio.intel.models.IntelSummary` carries
  ``is_model_written: Literal[True]``, so a renderer cannot present it as framework prose.
* It is cited. Every summary must carry at least one citation pointing at a page the
  framework actually retrieved. A summary of "what is publicly known" with no citation is
  the model's prior, not intelligence, and this module refuses to emit one.
* It is bounded. Three to five sentences, a hard character cap, and a cap on citations.
* It is filtered. The prose is the output of a model that just read hostile pages, and it
  ends up in front of a human. A sentence in it shaped like an instruction did not come
  from the researcher's judgement -- it came from a page -- so it is dropped and counted.

None of this makes the summary evidence. It moves no number: the numbers come from phase
two, under the influence budget. The summary explains them.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Sequence

from vulnprio.intel.models import (
    MAX_CITATIONS,
    MAX_SUMMARY_CHARS,
    IntelCitation,
    IntelConfig,
    IntelDocument,
    IntelGather,
    IntelSummary,
    IntelUsage,
    utc_now,
)
from vulnprio.intel.queries import normalize_url
from vulnprio.llm.heuristic import strip_imperative_sentences

__all__ = ["split_sentences", "select_citations", "summarize_intel"]

#: A sentence runs to a terminator followed by whitespace, or to the end of the text.
#: Deliberately the same shape as the one in :mod:`vulnprio.llm.heuristic`, so "2.5.22"
#: and "exploit-db.com" do not split a sentence in half.
_SENTENCE_RE = re.compile(r"[^\n\r]+?(?:[.!?]+(?=\s)|$)", re.MULTILINE)


def split_sentences(text: str) -> list[str]:
    """Sentences of ``text``, stripped, with empties dropped."""
    out: list[str] = []
    for match in _SENTENCE_RE.finditer(text or ""):
        sentence = match.group(0).strip()
        if sentence:
            out.append(sentence)
    return out


def select_citations(
    citations: Iterable[IntelCitation],
    documents: Sequence[IntelDocument],
    limit: int = MAX_CITATIONS,
) -> tuple[IntelCitation, ...]:
    """Keep only citations that point at a page this run actually retrieved.

    A citation to a URL that is not among the documents cannot be checked by anyone
    reading the report: the framework has no copy of what it said. Dropping it is the
    difference between a citation and a footnote-shaped assertion.
    """
    retrieved = {normalize_url(document.url) for document in documents}
    retrieved.discard("")
    kept: list[IntelCitation] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        key = normalize_url(citation.url)
        if not key or key not in retrieved:
            continue
        identity = (key, citation.cited_text[:80])
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(citation)
        if len(kept) >= limit:
            break
    return tuple(kept)


def summarize_intel(
    gathered: IntelGather,
    documents: Sequence[IntelDocument] | None = None,
    *,
    config: IntelConfig | None = None,
    model: str = "",
    prompt_hash: str = "",
    usage: IntelUsage | None = None,
    recorded: bool = False,
    now: datetime | None = None,
) -> IntelSummary | None:
    """Build the cited summary, or return ``None`` when no honest one can be built.

    ``None`` is returned -- never a placeholder and never an uncited summary -- when the
    researcher call produced no prose, when nothing it said can be tied to a retrieved
    page, or when everything it said was instruction-shaped. The caller reports the
    absence; :class:`IntelSummary` itself requires at least one citation at the type level,
    so an uncited summary is not merely refused here but unrepresentable anywhere.
    """
    config = config or IntelConfig()
    documents = tuple(documents if documents is not None else gathered.documents)

    text = (gathered.narrative or "").strip()
    if not text:
        return None

    citations = select_citations(gathered.citations, documents)
    if not citations:
        return None

    # The prose came from a model that just read pages written by strangers. Anything in
    # it shaped like a directive is relayed payload, not research.
    kept_sentences, dropped = strip_imperative_sentences(text)
    if not kept_sentences:
        return None

    sentences = split_sentences(" ".join(kept_sentences))
    if not sentences:
        return None
    # Trim the tail rather than the head: a researcher's first sentences carry the finding.
    # Brevity is not trimmed back up -- a two-sentence honest answer is better than a
    # padded one, and the sentence count is a readability rule, not a security property.
    sentences = sentences[: max(1, config.max_summary_sentences)]

    body = " ".join(sentences).strip()
    if len(body) > MAX_SUMMARY_CHARS:
        body = body[:MAX_SUMMARY_CHARS].rstrip()
    if not body:
        return None

    if dropped:
        note = f" [{dropped} instruction-shaped sentence(s) removed by the sandbox]"
        if len(body) + len(note) <= MAX_SUMMARY_CHARS:
            body += note

    return IntelSummary(
        text=body,
        citations=citations,
        model=model or gathered.model,
        prompt_hash=prompt_hash,
        usage=usage if usage is not None else gathered.usage,
        generated_at=now or utc_now(),
        recorded=bool(recorded),
    )
