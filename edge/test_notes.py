"""edge/NOTES.md against the tracked runs/*.json it cites (stdlib only).

Every number written as `<value>[ ms] (<file>.json › key/path)` must equal the JSON value
rounded to the decimals shown. A citation with no number in front of it fails too.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
CITE = re.compile(r"\((\w+\.json) › ([^)]+)\)")
NUM_CITE = re.compile(r"(\d[\d,]*(?:\.\d+)?)(?: ms)? \((\w+\.json) › ([^)]+)\)")


def lookup(fname, path):
    v = json.load(open(os.path.join(RUNS, fname)))
    for k in path.split("/"):
        v = v[k]
    return v


def main(notes=os.path.join(HERE, "NOTES.md")):
    text = open(notes, encoding="utf-8").read()
    cites, checked, bad = len(CITE.findall(text)), 0, []
    for num, fname, path in NUM_CITE.findall(text):
        dec = len(num.split(".")[1]) if "." in num else 0
        want = round(float(lookup(fname, path)), dec)
        if float(num.replace(",", "")) != want:
            bad.append(f"{num} ({fname} › {path}) but JSON gives {want}")
        checked += 1
    if checked != cites:
        bad.append(f"{cites - checked} citation(s) without a number in front")
    if checked < 30:
        bad.append(f"only {checked} cited numbers")
    for b in bad:
        print("FAIL", b)
    print(f"{checked} cited numbers checked, {len(bad)} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
