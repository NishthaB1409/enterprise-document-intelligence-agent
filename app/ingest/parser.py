"""PDF bytes to text, one entry per page.

Pages stay separate all the way through chunking, so every chunk carries an
exact page number. A citation that says "page 7" has to actually be page 7 — if
a chunk were allowed to span a page boundary, its page number would be a guess.
"""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class UnreadableDocument(Exception):
    """The upload is not a PDF this service can extract text from.

    Raised for corrupt files, password-protected files, and scans with no text
    layer. Callers turn this into a 4xx: it is the uploader's problem to fix,
    not a server fault.
    """


@dataclass(frozen=True, slots=True)
class Page:
    # 1-based, so it matches what a reader sees in a PDF viewer.
    number: int
    text: str


# A line break mid-word, from a word hyphenated across lines during typesetting.
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
# Two or more newlines: a real paragraph break, worth keeping.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n\s*")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")


def normalize(text: str) -> str:
    """Undo PDF line-wrapping without destroying paragraph structure.

    Extracted PDF text is hard-wrapped at the visual line width, so a sentence
    arrives full of newlines that mean nothing semantically. Left alone they
    fragment the text the splitter and the embedding model see. Blank lines are
    the one break worth preserving — they mark a real paragraph boundary, which
    is where the splitter prefers to cut.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _PARAGRAPH_BREAK.sub("\n\n", text)
    # Single newlines are line-wrap artifacts, not meaning.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    return text.strip()


def parse_pdf(data: bytes) -> list[Page]:
    """Extract per-page text. Pages with no text layer are dropped, but the
    numbering of the pages that remain still matches the original document."""
    try:
        reader = PdfReader(io.BytesIO(data))
        # Checked before touching pages: an encrypted reader raises on access,
        # and "wrong password" deserves a clearer message than a parse error.
        if reader.is_encrypted:
            raise UnreadableDocument(
                "the PDF is password-protected; decrypt it before uploading"
            )
        raw = [page.extract_text() or "" for page in reader.pages]
    except UnreadableDocument:
        raise
    except (PdfReadError, OSError, ValueError, KeyError) as exc:
        raise UnreadableDocument(f"could not read the PDF: {exc}") from exc

    pages = [
        Page(number=number, text=normalized)
        for number, text in enumerate(raw, start=1)
        if (normalized := normalize(text))
    ]

    if not pages:
        # Almost always a scan. Say so, rather than silently indexing nothing and
        # letting the user discover later that the document answers no questions.
        raise UnreadableDocument(
            "no extractable text found; the PDF appears to be a scan and needs OCR first"
        )

    return pages
