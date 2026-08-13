"""Build PDFs in memory, so ingestion tests exercise the real parser.

Checking a fixture PDF into the repo would work too, but a builder makes the
input to each test visible in the test itself — which page a sentence lands on
is usually the thing being asserted.
"""

import io
from collections.abc import Sequence

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen.canvas import Canvas

_LEFT_MARGIN = 72
_TOP = 750
_LINE_HEIGHT = 14


def build_pdf(pages: Sequence[str]) -> bytes:
    """One entry per page. An empty string produces a page with no text layer —
    what a scan looks like to the parser."""
    buffer = io.BytesIO()
    # `invariant` freezes the creation date and document id that reportlab
    # otherwise stamps into every file. Without it the same text produces
    # different bytes on each call, and `document_id` — a hash of the bytes —
    # would look non-deterministic when it is the builder that isn't.
    canvas = Canvas(buffer, pagesize=LETTER, invariant=1)

    for page in pages:
        cursor = _TOP
        for line in page.splitlines():
            if line.strip():
                canvas.drawString(_LEFT_MARGIN, cursor, line)
            cursor -= _LINE_HEIGHT
        canvas.showPage()

    canvas.save()
    return buffer.getvalue()
