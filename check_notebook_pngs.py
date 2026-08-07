#!/usr/bin/env python3
"""
Validate the PNG images stored inside notebook code-cell outputs.

Why this matters: xdvipdfmx (the XeLaTeX -> PDF stage) uses libpng, which is
strict about CRCs. A single corrupt stored plot aborts PDF writing with
    libpng error: bad adaptive filter value
    xdvipdfmx: Command for 'xdvipdfmx' gave return code 6
and the resulting PDF is unopenable, even though LaTeX typeset every page.
Browsers are lenient about the same corruption, so the HTML site looks fine.

Fix for anything reported below: re-run that notebook (or just that cell) so
the output is regenerated, then re-export.

Usage:
    python3 check_notebook_pngs.py            # all chap_*.ipynb
    python3 check_notebook_pngs.py chap_07_trees.ipynb
"""
import base64
import glob
import json
import struct
import sys
import zlib


def png_problem(raw):
    """Return None if the PNG is well formed, else a short reason."""
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return "not a PNG (bad magic)"
    i = 8
    while i < len(raw) - 8:
        length = struct.unpack(">I", raw[i:i + 4])[0]
        ctype = raw[i + 4:i + 8]
        if i + 12 + length > len(raw):
            return "truncated chunk %s" % ctype.decode("latin1", "replace")
        payload = raw[i + 8:i + 8 + length]
        stored = struct.unpack(">I", raw[i + 8 + length:i + 12 + length])[0]
        if zlib.crc32(ctype + payload) & 0xFFFFFFFF != stored:
            return "bad CRC in %s" % ctype.decode("latin1", "replace")
        i += 12 + length
        if ctype == b"IEND":
            return None
    return "no IEND chunk"


def check(path):
    nb = json.load(open(path, encoding="utf-8"))
    total, bad = 0, []
    for index, cell in enumerate(nb.get("cells", [])):
        for output in cell.get("outputs", []):
            b64 = (output.get("data") or {}).get("image/png")
            if not b64:
                continue
            if isinstance(b64, list):
                b64 = "".join(b64)
            total += 1
            try:
                raw = base64.b64decode(b64)
            except Exception as exc:
                bad.append((index, "undecodable base64: %s" % exc))
                continue
            problem = png_problem(raw)
            if problem:
                bad.append((index, problem))
    return total, bad


def main():
    targets = sys.argv[1:] or sorted(glob.glob("chap_*.ipynb"))
    grand_total, grand_bad = 0, 0
    for path in targets:
        total, bad = check(path)
        grand_total += total
        grand_bad += len(bad)
        if bad:
            print("%-40s %d/%d CORRUPT" % (path, len(bad), total))
            for index, reason in bad:
                print("      cell %-5d %s" % (index, reason))
        elif total:
            print("%-40s %d ok" % (path, total))
    print("\n%d stored PNG outputs, %d corrupt." % (grand_total, grand_bad))
    if grand_bad:
        print("Re-run the notebooks listed above, then re-export.")
    return 1 if grand_bad else 0


if __name__ == "__main__":
    sys.exit(main())
