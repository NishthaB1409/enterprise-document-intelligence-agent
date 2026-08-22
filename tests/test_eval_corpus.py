"""Integrity of the evaluation ground truth.

The numbers in `eval/README.md` rest on one assumption: that a gold span
identifies exactly one passage, in the document it is supposed to be in. Nothing
enforces that at runtime — the harness would happily score against a corpus
where a distractor had drifted into answering a question, and the result would
be wrong in a direction nobody would notice, because it would look like a
retrieval improvement.

So the assumption is a test. These are cheap, they need no models, and they fail
the build rather than quietly invalidating a published measurement.
"""

import re

import pytest

from app.ingest.chunking import build_splitter, chunk_pages
from eval.corpus import CORPUS, GOLD_DOCUMENTS
from eval.distractors import DISTRACTORS
from eval.metrics import is_relevant, normalize
from eval.questions import QUESTIONS
from eval.retrieval_eval import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, TOP_K


def _pages():
    return [(doc, page) for doc in CORPUS for page in doc.pages]


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q.doc_id)
def test_every_gold_span_appears_exactly_once_in_the_corpus(question):
    matches = [
        (doc.doc_id, page.number)
        for doc, page in _pages()
        if normalize(question.gold_span) in normalize(page.text)
    ]

    assert len(matches) == 1, (
        f"{question.question!r} has {len(matches)} matching passages ({matches}); "
        "a gold span must identify exactly one, or relevance is ambiguous"
    )
    assert matches[0][0] == question.doc_id


def test_no_distractor_answers_a_question():
    """Stated separately from the uniqueness check because it is the failure
    that matters most and the one most likely to be introduced later: adding a
    distractor is easy, and noticing it happens to contain an answer is not."""
    leaks = [
        (question.question, doc.doc_id)
        for question in QUESTIONS
        for doc in DISTRACTORS
        for page in doc.pages
        if normalize(question.gold_span) in normalize(page.text)
    ]

    assert not leaks, f"distractors containing gold answers: {leaks}"


def test_gold_spans_survive_chunking():
    """A span split across a chunk boundary is unreachable: no single chunk
    contains it, so every configuration scores zero on that question and the
    corpus quietly gets harder in a way that looks like a model regression."""
    splitter = build_splitter(CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    chunks = [
        chunk
        for document in CORPUS
        for chunk in chunk_pages(document.pages, document.doc_id, splitter)
    ]

    unreachable = [
        question.question
        for question in QUESTIONS
        if not any(normalize(question.gold_span) in normalize(c.text) for c in chunks)
    ]

    assert not unreachable, (
        f"gold spans no chunk contains: {unreachable}. "
        "Chunk settings changed, or the span crosses a boundary."
    )


def test_the_question_set_is_balanced_across_kinds():
    """The report breaks results down by kind. A slice of two questions would
    produce a number that moves 0.5 on a single rank change and reads as a
    finding."""
    kinds = {kind: sum(q.kind == kind for q in QUESTIONS) for kind in ("lexical", "semantic")}

    assert min(kinds.values()) >= 5, f"too few questions in a slice to report on: {kinds}"


def test_the_corpus_is_large_enough_for_top_k_to_discriminate():
    """Guards the mistake this eval was first written with.

    With ten chunks in the corpus, retrieving five of them returns half the
    index: every configuration scores near the ceiling, the differences are one
    question wide, and the comparison cannot support a conclusion in either
    direction. The distractors exist to fix that, and this stops them being
    deleted as dead weight.
    """
    splitter = build_splitter(CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    total = sum(
        len(chunk_pages(document.pages, document.doc_id, splitter)) for document in CORPUS
    )

    assert total >= TOP_K * 4, (
        f"{total} chunks against top_k={TOP_K} leaves too little room to rank; "
        "add documents to eval/distractors.py"
    )


def test_relevance_ignores_whitespace_and_case():
    """The splitter reflows line breaks, so a gold span copied from the corpus
    will not match a chunk byte-for-byte. If this stopped holding, every
    question would score zero and it would look like retrieval had broken."""
    from app.ingest.chunking import Chunk
    from app.vectorstore.store import ScoredChunk

    chunk = ScoredChunk(
        chunk=Chunk(
            id="c",
            doc_id="d",
            index=0,
            page=1,
            text="Either party may TERMINATE\n   this Agreement\tfor convenience.",
            char_start=0,
            char_end=0,
        ),
        source="s.pdf",
        score=1.0,
    )

    assert is_relevant(chunk, "terminate this Agreement for convenience")


def test_gold_documents_and_distractors_do_not_overlap():
    gold = {doc.doc_id for doc in GOLD_DOCUMENTS}
    distractor = {doc.doc_id for doc in DISTRACTORS}

    assert not gold & distractor
    assert len(CORPUS) == len(gold) + len(distractor)


def test_document_ids_are_unique():
    """Two documents sharing an id would collide on chunk ids, and the second
    would silently overwrite the first at ingest."""
    ids = [document.doc_id for document in CORPUS]

    assert len(ids) == len(set(ids)), f"duplicate doc_ids: {ids}"


def test_sources_look_like_filenames():
    """`source` is what a citation shows the reviewer, so it has to read as a
    document, not an internal key."""
    for document in CORPUS:
        assert re.fullmatch(r"[a-z0-9][a-z0-9.\-]*\.pdf", document.source), document.source
