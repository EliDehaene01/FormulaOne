"""
Splits knowledge_base/*.md into retrievable chunks, one per ## section.

## headers are the natural chunk boundary here: each knowledge_base file is
hand-written with one self-contained concept per section (see
backend/knowledge_base/*.md) — a whole-file chunk would mix several
unrelated concepts into a single embedding (hurts recall: the vector ends
up an average of everything in the file, not close to any one concept),
while splitting smaller than a section would cut a concept's explanation
in half. The # document title is prepended to every chunk's text so the
embedding still carries "this is about tyre strategy" context even for a
short section that doesn't repeat it.
"""

from pathlib import Path

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"


def chunk_markdown(text: str, source: str) -> list[dict]:
    """One dict per ## section: {"text", "source", "section"}. `source` is the filename, `section` is that header's text (used later as retrieval metadata so a result can point back to exactly where it came from)."""
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        doc_title = lines[0][2:].strip()
        lines = lines[1:]
    else:
        doc_title = source

    chunks: list[dict] = []
    section_header: str | None = None
    section_lines: list[str] = []

    def _flush() -> None:
        body = "\n".join(section_lines).strip()
        if not body:
            return
        header = section_header or doc_title
        heading = f"## {header}" if section_header else ""
        text_block = "\n\n".join(part for part in (f"# {doc_title}", heading, body) if part)
        chunks.append({"text": text_block, "source": source, "section": header})

    for line in lines:
        if line.startswith("## "):
            _flush()
            section_header = line[3:].strip()
            section_lines = []
        else:
            section_lines.append(line)
    _flush()  # the last section never hits the "## " branch to trigger its own flush

    return chunks


def chunk_knowledge_base(directory: Path = KNOWLEDGE_BASE_DIR) -> list[dict]:
    """Every ## section from every .md file in `directory`, in filename order."""
    all_chunks: list[dict] = []
    for path in sorted(directory.glob("*.md")):
        all_chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=path.name))
    return all_chunks
