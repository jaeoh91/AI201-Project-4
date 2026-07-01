"""One-off script: fetch ~10 genuine human-written calibration samples from
Project Gutenberg (public-domain, pre-LLM-era text — unambiguously human)
and save them to fixtures/human/*.txt.

Deliberately spans registers: narrative fiction, personal letters, and
formal essays/political writing (the last to exercise the known
false-positive blind spot in planning.md §5 — formulaic human prose that
legitimately scores "AI-like" on these heuristics).

Usage: .venv/bin/python fixtures/fetch_human_samples.py
"""

import os
import re
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human")

# (filename, Gutenberg plaintext URL, author/work, target paragraph index
#  among paragraphs of ~80-260 words found after the Gutenberg header)
SOURCES = [
    ("human_01_austen.txt", "https://www.gutenberg.org/files/1342/1342-0.txt", "Jane Austen, Pride and Prejudice", 8),
    ("human_02_doyle.txt", "https://www.gutenberg.org/files/1661/1661-0.txt", "Arthur Conan Doyle, The Adventures of Sherlock Holmes", 10),
    ("human_03_melville.txt", "https://www.gutenberg.org/files/2701/2701-0.txt", "Herman Melville, Moby-Dick", 15),
    ("human_04_twain.txt", "https://www.gutenberg.org/files/76/76-0.txt", "Mark Twain, Adventures of Huckleberry Finn", 12),
    ("human_05_carroll.txt", "https://www.gutenberg.org/files/11/11-0.txt", "Lewis Carroll, Alice's Adventures in Wonderland", 6),
    ("human_06_wells.txt", "https://www.gutenberg.org/files/36/36-0.txt", "H. G. Wells, The War of the Worlds", 20),
    ("human_07_emerson_formal.txt", "https://www.gutenberg.org/files/2944/2944-0.txt", "Ralph Waldo Emerson, Essays, First Series (formal register)", 25),
    ("human_08_federalist_formal.txt", "https://www.gutenberg.org/files/1404/1404-0.txt", "Publius, The Federalist Papers (formal/political register)", 30),
    ("human_09_thoreau.txt", "https://www.gutenberg.org/files/205/205-0.txt", "Henry David Thoreau, Walden", 18),
    ("human_10_shelley.txt", "https://www.gutenberg.org/files/84/84-0.txt", "Mary Shelley, Frankenstein", 14),
]

HEADER_MARK = "*** START OF"
FOOTER_MARK = "*** END OF"


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def strip_gutenberg_boilerplate(raw):
    start = raw.find(HEADER_MARK)
    if start != -1:
        start = raw.find("\n", start) + 1
    else:
        start = 0
    end = raw.find(FOOTER_MARK)
    if end == -1:
        end = len(raw)
    return raw[start:end]


def extract_paragraphs(body):
    # Gutenberg plaintext wraps lines; paragraphs are separated by blank lines.
    raw_paragraphs = re.split(r"\n\s*\n", body)
    cleaned = []
    for p in raw_paragraphs:
        p = " ".join(line.strip() for line in p.splitlines()).strip()
        word_count = len(p.split())
        sentence_count = len(re.findall(r"[.!?]+", p))
        if 90 <= word_count <= 260 and sentence_count >= 3 and not p.isupper():
            cleaned.append(p)
    return cleaned


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_lines = []
    for filename, url, source, target_index in SOURCES:
        raw = fetch_text(url)
        body = strip_gutenberg_boilerplate(raw)
        paragraphs = extract_paragraphs(body)
        if not paragraphs:
            print(f"!! no qualifying paragraph found for {source}")
            continue
        index = min(target_index, len(paragraphs) - 1)
        paragraph = paragraphs[index]

        # File contains ONLY the paragraph text — no header — so it scores
        # cleanly; provenance is tracked separately in SOURCES.txt.
        out_path = os.path.join(OUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(paragraph + "\n")
        manifest_lines.append(f"{filename}: {source} ({url})")
        print(f"wrote {out_path} ({len(paragraph.split())} words) <- {source}")

    with open(os.path.join(OUT_DIR, "SOURCES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")


if __name__ == "__main__":
    main()
