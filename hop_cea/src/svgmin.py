"""Shrink matplotlib SVGs for Notion attachments.

Rounds coordinates inside geometry attributes only (never inside text), drops
metadata and comments, and merges whitespace, so a plot fits comfortably under
Notion's 200 KiB inline-attachment limit.
"""
import re, glob, os

NUM = re.compile(r'-?\d+\.\d+')
GEOM = re.compile(r'\b(d|points|x|y|x1|x2|y1|y2|width|height|transform|viewBox|r|cx|cy)="([^"]*)"')


def _round(m):
    v = ('%.1f' % float(m.group(0))).rstrip('0').rstrip('.')
    return '0' if v == '-0' else v


def shrink(s):
    s = re.sub(r'<metadata>.*?</metadata>', '', s, flags=re.S)
    s = re.sub(r'<!--.*?-->', '', s, flags=re.S)
    s = re.sub(r'<\?xml.*?\?>\s*<!DOCTYPE.*?>', '', s, flags=re.S)
    s = GEOM.sub(lambda m: f'{m.group(1)}="{NUM.sub(_round, m.group(2))}"', s)
    s = re.sub(r'\n\s*', '\n', s)
    return s.strip()


if __name__ == "__main__":
    os.makedirs('out/min', exist_ok=True)
    for f in sorted(glob.glob('out/*.svg')):
        t = shrink(open(f).read())
        open('out/min/' + os.path.basename(f), 'w').write(t)
        print(os.path.basename(f), len(t))
