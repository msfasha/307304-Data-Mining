#!/usr/bin/env python3
"""Build the 20261 Data Mining course website: stage sources -> quarto render -> zip bundles.

Run from anywhere:  python3 20261/site/build.py
Output: 20261/_build/site/   (never modifies the teaching sources)

Modules are auto-discovered: every folder named "Module <N>-<Title>" under 20261/ becomes a module.
Nothing here depends on anything outside this repository.
"""
import re, shutil, subprocess, sys, zipfile
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
SRC = HERE.parent                      # 20261/
REPO = SRC.parent                      # repository root
DATASETS = REPO / "datasets"
STAGE = SRC / "_build" / "stage"
OUT = SRC / "_build" / "site"
REPO_URL = "https://github.com/msfasha/307304-Data-Mining"
SEMESTER = "First semester 2025/2026 (20261)"
TITLE_FIXES = {"Associatin": "Association"}   # typos in folder names, fixed for display only
IMG = (".png", ".jpg", ".jpeg", ".gif", ".svg")
DATA = (".csv", ".txt", ".xlsx", ".json")

def safe(name):                        # staged file names: no spaces, so URLs stay simple
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def modules():
    found = []
    for p in SRC.iterdir():
        m = re.match(r"Module (\d+(?:\.\d+)?)\s*-\s*(.+)$", p.name)
        if p.is_dir() and m:
            title = m.group(2).strip()
            for a, b in TITLE_FIXES.items():
                title = title.replace(a, b)
            found.append((float(m.group(1)), m.group(1), title, p))
    return sorted(found)

def md_page(src, dst, title):
    text = re.sub(r"\A# .*\n", "", src.read_text(encoding="utf-8"), count=1)
    dst.write_text(f'---\ntitle: "{title}"\ntoc: true\n---\n\n{text}', encoding="utf-8")

def main():
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    shutil.copy(HERE / "_quarto.yml", STAGE / "_quarto.yml")
    shutil.copy(HERE / "styles.css", STAGE / "styles.css")
    nb_bundle, pdf_bundle, cards = [], [], []

    for num, numtxt, title, s in modules():
        slug = f"module-{numtxt.replace('.', '-')}"
        d = STAGE / slug
        d.mkdir()
        links = []   # (label, href)

        def add_files(folder, sub, kind_pdf, kind_nb, bundle_pdf):
            """Copy PDFs / notebooks / markdown / data of one folder (module root or a subfolder)."""
            out = d / sub if sub else d
            out.mkdir(exist_ok=True)
            pre = f"{sub}/" if sub else ""
            for p in sorted(folder.glob("*.pdf")):
                shutil.copy(p, out / safe(p.name))
                links.append((f"{kind_pdf}: {p.stem}", pre + safe(p.name)))
                if bundle_pdf:
                    pdf_bundle.append((p, f"{slug}/{pre}{p.name}"))
            for p in sorted(folder.glob("*.ipynb")):
                shutil.copy(p, out / safe(p.name))
                st = Path(safe(p.name)).stem
                links.append((f"{kind_nb} (HTML): {p.stem}", pre + st + ".html"))
                links.append((f"{kind_nb} notebook (.ipynb): {p.stem}", pre + safe(p.name)))
                nb_bundle.append((p, f"{slug}/{pre}{p.name}"))
            for p in sorted(folder.glob("*.md")):
                md_page(p, out / (Path(safe(p.name)).stem + ".qmd"), p.stem)
                links.append((f"Reading (HTML): {p.stem}", pre + Path(safe(p.name)).stem + ".html"))
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG + DATA:
                    shutil.copy(p, out / safe(p.name))
                    nb_bundle.append((p, f"{slug}/{pre}{p.name}"))

        add_files(s, "", "Slides (PDF)", "Lab", True)
        for sub, kp, kn in (("additional", "Additional reading (PDF)", "Additional lab"),
                            ("optional", "Optional reading (PDF)", "Optional lab")):
            if (s / sub).is_dir():
                add_files(s / sub, sub, kp, kn, True)
        cards.append((slug, numtxt, title, links))

    for slug, numtxt, title, links in cards:
        rows = "\n".join(f"- [{l}]({quote(h)})" + (" - (optional)" if l.startswith("Optional") else "")
                         for l, h in links)
        (STAGE / slug / "index.qmd").write_text(f'---\ntitle: "Module {numtxt} - {title}"\n---\n\n{rows}\n',
                                                encoding="utf-8")
    body = ["---", 'title: "307304 Data Mining"', f'subtitle: "{SEMESTER}"', "toc: false", "---", "",
            "## Modules", ""]
    body += [f"- [Module {n} - **{t}**]({slug}/index.qmd)" for slug, n, t, _ in cards]
    body += ["", "## Downloads", "",
             "- [Download all labs and datasets (.zip)](downloads/datamining-labs-notebooks.zip)",
             "- [Download all PDFs: slides and readings (.zip)](downloads/datamining-course-pdfs.zip)",
             "- [Datasets (.zip)](downloads/datamining-datasets.zip)",
             f"- [Source on GitHub]({REPO_URL})", ""]
    (STAGE / "index.qmd").write_text("\n".join(body), encoding="utf-8")

    # zips go in the stage so quarto copies them as resources
    dl = STAGE / "downloads"
    dl.mkdir()
    datasets = [(p, f"datasets/{p.name}") for p in sorted(DATASETS.glob("*")) if p.is_file()] if DATASETS.is_dir() else []
    readme = ("Run labs in Jupyter / VS Code / Colab. One folder per module.\n"
              "Notebooks load their datasets from GitHub; local copies are in datasets/.\n")
    for name, files in (("datamining-labs-notebooks.zip", nb_bundle + datasets),
                        ("datamining-course-pdfs.zip", pdf_bundle),
                        ("datamining-datasets.zip", datasets)):
        with zipfile.ZipFile(dl / name, "w", zipfile.ZIP_DEFLATED) as z:
            for p, arc in files:
                z.write(p, arc)
            if "notebooks" in name:
                z.writestr("README.txt", readme)

    subprocess.run(["quarto", "render"], cwd=STAGE, check=True)
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.move(str(STAGE / "_site"), OUT)
    print("Built ->", OUT)

if __name__ == "__main__":
    sys.exit(main())
