#!/usr/bin/env python3
"""
Strip terminal control characters from notebook text outputs.

Keras/tqdm progress bars redraw themselves with backspaces (0x08), carriage
returns and ANSI escape sequences. Those bytes are meaningless once the output
is stored in a .ipynb, but they are fatal to the PDF build: XeLaTeX reports

    ! Text line contains an invalid character.
    l.754 .../step^^H^^H^^H^^H...

and after 100 such errors LaTeX gives up mid-document, silently truncating the
book (we lost chapters 9 onward this way).

This rewrites stream and text/plain outputs only: each progress bar collapses
to its final state, exactly what a reader should see. Images, code, markdown
and execution counts are untouched.

Everything here has to be right for **both** outputs, because the notebooks feed
the HTML site as well as the PDF. Print-only repairs belong in
`fix_tex_verbatim.py`, which acts on the LaTeX export and cannot reach the site.

Usage:
    python3 clean_notebook_output.py                    # report only
    python3 clean_notebook_output.py --apply            # rewrite
    python3 clean_notebook_output.py --apply chap_08*.ipynb
"""
import glob
import json
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def collapse(text):
    """Apply carriage returns and backspaces, then drop leftover control bytes."""
    text = ANSI.sub("", text)
    out = []
    for line in text.split("\n"):
        buf = []
        for ch in line:
            if ch == "\r":
                buf = []
            elif ch == "\b":
                if buf:
                    buf.pop()
            else:
                buf.append(ch)
        out.append("".join(buf))
    return CTRL.sub("", "\n".join(out))


def clean_output(output):
    changed = plain_rich_output(output)
    if output.get("output_type") == "stream" and "text" in output:
        raw = output["text"]
        joined = "".join(raw) if isinstance(raw, list) else raw
        new = collapse(joined)
        if new != joined:
            output["text"] = new.splitlines(keepends=True)
            changed = True
    data = output.get("data") or {}
    if "text/plain" in data:
        raw = data["text/plain"]
        joined = "".join(raw) if isinstance(raw, list) else raw
        new = collapse(joined)
        if new != joined:
            data["text/plain"] = new.splitlines(keepends=True)
            changed = True
    return changed


def plain_rich_output(output):
    """Drop the scikit-learn estimator diagram, which helps neither output.

    A fitted estimator stores a 20 kB text/html diagram alongside a one-line
    text/plain repr. MyST uses that HTML for the website *and* the PDF, and it
    carries the line "In a Jupyter environment, please rerun this cell to show
    the HTML repr" - meaningless on a static site, absurd in print.

    Notebooks now set `sklearn.set_config(display='text')`, so nothing new is
    stored this way; this clears what predates that setting.

    Note what is deliberately *not* dropped: the rich <pre> that Keras
    `model.summary()` produces. Its styling is worth having on the website, and
    print gets what it needs from `fix_tex_verbatim.py` instead. Nor are pandas
    DataFrames, whose HTML is exactly what both outputs want.
    """
    data = output.get("data") or {}
    html = "".join(data.get("text/html") or [])
    if not html or "text/plain" not in data:
        return False
    if "sk-container" in html:
        del data["text/html"]
        return True
    return False


PROGRESS = re.compile(r"^\s*(?:\d+/\d+\s|Epoch \d+/\d+)")


def collapse_progress(cell):
    """Keep only the last line of a training progress run.

    Keras emits one stream output per progress-bar refresh, so a single fit()
    can leave several hundred outputs behind. In the PDF each one becomes its
    own shaded panel, filling page after page with `447/447 - loss: ...`. A
    reader only needs where training ended, so consecutive progress outputs
    collapse to the final one.
    """
    outputs = cell.get("outputs", [])
    if not outputs:
        return 0
    kept, run, removed = [], [], 0
    for output in outputs:
        text = "".join(output.get("text", [])) if output.get("output_type") == "stream" else ""
        if text and PROGRESS.match(text):
            run.append(output)
            continue
        if run:
            kept.append(run[-1]); removed += len(run) - 1; run = []
        kept.append(output)
    if run:
        kept.append(run[-1]); removed += len(run) - 1
    if removed:
        cell["outputs"] = kept
    return removed


def drop_blank_outputs(cell):
    """Remove outputs that carry nothing but whitespace.

    MyST still emits a `verbatim` for these, which in print is an empty shaded
    panel sitting between two paragraphs. They come from rich writing a closing
    <pre></pre> after a progress display.
    """
    outputs = cell.get("outputs", [])
    kept = []
    for output in outputs:
        data = output.get("data") or {}
        if set(data) == {"text/plain"} and not "".join(data["text/plain"]).strip():
            continue
        if output.get("output_type") == "stream" and not "".join(output.get("text") or []).strip():
            continue
        kept.append(output)
    removed = len(outputs) - len(kept)
    if removed:
        cell["outputs"] = kept
    return removed


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply_changes = "--apply" in sys.argv
    do_progress = "--collapse-progress" in sys.argv
    paths = args or sorted(glob.glob("chap_*.ipynb"))

    total = 0
    for path in paths:
        nb = json.load(open(path, encoding="utf-8"))
        touched = 0
        for cell in nb.get("cells", []):
            if do_progress:
                dropped = collapse_progress(cell)
                if dropped:
                    touched += dropped
            touched += drop_blank_outputs(cell)
            for output in cell.get("outputs", []):
                if clean_output(output):
                    touched += 1
        if touched:
            total += touched
            print("%-40s %d output(s) cleaned" % (path, touched))
            if apply_changes:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(nb, handle, ensure_ascii=False, indent=1)
                    handle.write("\n")
    if not total:
        print("nothing to clean")
    elif not apply_changes:
        print("\n(dry run - pass --apply to rewrite)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
