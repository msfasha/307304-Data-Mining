# Publishing the course website (20261)

Students use <https://msfasha.github.io/307304-Data-Mining/>.
Only the current semester folder (`20261/`) is published. Older semesters are not.

This repository is fully self-contained: everything needed to rebuild and publish the site is in it
(`20261/site/`, `.github/workflows/publish-course.yml`, `datasets/`). No other folder or repo is needed.

## Routine: how to publish a change

1. Edit teaching files under `20261/Module N-.../` (and `datasets/` if needed).
2. (Optional) preview: `python3 20261/site/build.py`, then open `20261/_build/site/index.html`.
   Needs Python 3 and [Quarto](https://quarto.org) (the workflow pins 1.11.9). No R, no Jupyter kernel needed.
3. `git add`, `git commit`, `git push` to `main`. GitHub Actions builds and deploys automatically
   (watch the Actions tab; it can also be started manually via Run workflow).

## How it works

- `20261/site/build.py` stages the teaching files, runs `quarto render`, and zips the downloads.
  It never edits the teaching sources. Output goes to `20261/_build/site/` (gitignored).
- `.github/workflows/publish-course.yml` runs the build and deploys to GitHub Pages on every push to `main`
  that touches `20261/**` or `datasets/**`.
- Quarto is only a notebook/Markdown-to-HTML converter here. No code is executed (`execute: enabled: false`),
  so labs show the outputs saved in the notebook. Run notebooks and save them before committing.

## One-time setup for a new repository (do this once)

- Repo Settings > Pages > Source = **GitHub Actions**.
- Set the repo "Website" field (gear next to About) to the site URL.
- Keep this link under the title of `README.md`:

```md
[Open the student course website](https://msfasha.github.io/307304-Data-Mining/) (current semester, 20261).
```

## What each module folder must contain

Modules are **auto-discovered**: any folder `20261/Module <N>-<Title>/` becomes a module page, ordered by N
(a decimal such as `2.1` works). Nothing to register. Each material is **one line**: `name - (HTML) (ipynb) (PDF)`.

| Location | Becomes |
|---|---|
| `*.ipynb` in the module folder | one line: (HTML) lab page, (ipynb) download, (PDF) **generated automatically** |
| `*.pdf` in the module folder | one line: name - (PDF) (hand-made, e.g. exported slides) |
| `additional/`, `optional/` | same rules; `optional/` lines are marked "(optional)" |
| `*.md` (module, additional, optional) | one line: (HTML) and (PDF) generated (first `# ` heading becomes the title) |
| `*.png/.jpg/.csv/.txt/.xlsx/.json` in the module folder | copied beside the notebooks (images used by notebooks, small data) |
| `datasets/*` (repo root) | "Datasets (.zip)" and included in the labs zip |

**PDFs are never committed.** The build prints every rendered notebook/markdown page to PDF with headless Chrome
(preinstalled on GitHub's `ubuntu-24.04` runners; locally `google-chrome` or `chromium` must be installed), so the
PDF always matches the notebook. Nothing to export or maintain by hand; just save the notebook with its outputs,
commit and push. All PDFs (generated and hand-made) are in "Download all PDFs (.zip)".

Notebook downloads are staged
under `_dl/` on purpose: Quarto rewrites any link to a rendered `.ipynb` into its `.html`, so the raw copies live
in a folder it does not render.

Not published: `*.pptx`/`*.docx` (gitignored), `beta/` folders (gitignored), folders starting with `skip_`,
and anything in other semester folders. File names with spaces are staged with underscores (downloads inside
the zips keep their original names). Typos in folder names can be corrected for display only in `TITLE_FIXES`
at the top of `build.py`.

Notebooks that load data use `https://raw.githubusercontent.com/msfasha/307304-Data-Mining/.../datasets/...`,
so the repo must stay public and `datasets/` must keep its path.

## Theme (keep this)

Petra burgundy `#7a1f3d` on Quarto's `cosmo` theme: headings and links burgundy, navbar burgundy with
**white** links, search icon included. All of it is in `20261/site/styles.css`. Homepage lists the modules,
then Downloads (labs + datasets zip, PDFs zip, datasets zip, GitHub link); each module has its own page of links.
Navbar title and homepage text are in `20261/site/_quarto.yml` and `build.py`.

## Starting a new semester (e.g. 20262)

1. Create the new `2026X/` folder with `Module N-Title` subfolders.
2. Copy `20261/site/` into it; in `build.py` update `SEMESTER` (and nothing else is path-bound: `SRC` is relative to the script).
3. In `.github/workflows/publish-course.yml` replace every `20261` with the new folder name.
4. In `.gitignore` update the `_build/` comment if wanted; update the README link text.

## Before you publish

Notebooks and notes become public. Check for API keys and personal data first (placeholders are fine).
