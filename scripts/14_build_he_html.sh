#!/usr/bin/env bash
# Render docs/he/*.md to RTL HTML in docs/he/html/. Needs pandoc. Private docs are skipped unless PRIVATE=1.
# Usage: scripts/14_build_he_html.sh [FILE.md ...]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/docs/he"; OUT="$SRC/html"; mkdir -p "$OUT"
command -v pandoc >/dev/null 2>&1 || { echo "pandoc not found (brew install pandoc)"; exit 1; }

CSS="$OUT/he.css"
cat > "$CSS" <<'EOF'
:root { --fg:#1b1b1b; --muted:#5a5a5a; --line:#d9d9d9; --accent:#1f6f3a; --code-bg:#f4f4f4; --th-bg:#e9f3ec; }
html { direction: rtl; }
body { font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, "Noto Sans Hebrew", sans-serif;
       color: var(--fg); background:#fff; max-width: 920px; margin: 0 auto; padding: 32px 24px 64px;
       line-height: 1.65; font-size: 16px; direction: rtl; text-align: right; }
h1, h2, h3, h4 { text-align: right; line-height: 1.3; }
h1 { font-size: 1.9em; color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: .3em; }
h2 { font-size: 1.4em; margin-top: 1.8em; border-bottom: 1px solid var(--line); padding-bottom: .2em; }
h3 { font-size: 1.15em; margin-top: 1.4em; }
p, li { text-align: right; }
ul, ol { padding-right: 1.6em; padding-left: 0; }
blockquote { margin: 1em 0; padding: .6em 1em; border-right: 4px solid var(--accent); border-left: 0; background: #f7faf8; color: #222; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; direction: rtl; font-size: .95em; }
th, td { border: 1px solid var(--line); padding: .45em .6em; text-align: right; vertical-align: top; }
th { background: var(--th-bg); }
tr:nth-child(even) td { background: #fafafa; }
/* code, commands, paths, identifiers stay LTR and isolated so they do not flip the line */
code, pre, kbd { font-family: ui-monospace, Menlo, Consolas, monospace; direction: ltr; unicode-bidi: isolate; }
code { background: var(--code-bg); padding: .1em .35em; border-radius: 4px; font-size: .9em; }
pre { background: var(--code-bg); padding: .8em 1em; border-radius: 6px; overflow-x: auto; text-align: left; line-height: 1.45; }
pre code { background: none; padding: 0; }
a { color: var(--accent); }
hr { border: 0; border-top: 1px solid var(--line); margin: 2em 0; }
strong { font-weight: 650; }
.subtitle, .muted { color: var(--muted); }
@media print {
  body { max-width: none; padding: 0; font-size: 11.5pt; }
  h2 { page-break-after: avoid; } table, pre, blockquote { page-break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
  @page { size: A4; margin: 18mm 16mm; }
}
EOF

build() {   # md file
  local md="$1"; local name; name="$(basename "$md" .md)"
  local title; title="$(grep -m1 '^# ' "$md" | sed 's/^# //')"
  pandoc "$md" -f gfm -t html5 --standalone --css he.css \
    --metadata title="$title" --metadata lang=he --metadata dir=rtl \
    -o "$OUT/$name.html"
  python3 - "$OUT/$name.html" <<'PY'
import re,sys
p=sys.argv[1]; s=open(p,encoding='utf-8').read()
s=re.sub(r'<header id="title-block-header">.*?</header>\s*', '', s, flags=re.S)
s=s.replace('<html xmlns="http://www.w3.org/1999/xhtml" lang="he" xml:lang="he" dir="rtl">','<html lang="he" dir="rtl">')
open(p,'w',encoding='utf-8').write(s)
PY
  echo "  $name.html"
}

PRIVATE_DOCS="SHARED_RESEARCH_AND_DECISIONS"
is_private() { case " $PRIVATE_DOCS " in *" $1 "*) return 0;; *) return 1;; esac; }

echo "building RTL HTML into docs/he/html/"
if [ $# -gt 0 ]; then for f in "$@"; do build "$SRC/$(basename "$f")"; done
else for f in "$SRC"/*.md; do n="$(basename "$f" .md)"; if is_private "$n" && [ -z "${PRIVATE:-}" ]; then echo "  (skipped private: $n.md)"; else build "$f"; fi; done; fi

{
  echo '<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><title>Nocturne — מסמכים בעברית</title><link rel="stylesheet" href="he.css"></head><body>'
  echo '<h1>Nocturne — מסמכים בעברית</h1><ul>'
  for f in "$OUT"/*.html; do n="$(basename "$f" .html)"; [ "$n" = index ] && continue; is_private "$n" && continue
    t="$(grep -m1 -o '<h1[^>]*>[^<]*' "$f" | sed 's/<h1[^>]*>//')"; echo "<li><a href=\"$n.html\">${t:-$n}</a> <span class=\"muted\">($n.md)</span></li>"; done
  echo '</ul><p class="muted">נוצר ע"י scripts/14_build_he_html.sh — לפתוח בדפדפן; להדפסה ל-PDF: ⌘P.</p></body></html>'
} > "$OUT/index.html"
echo "  index.html"
echo "open: $OUT/index.html"
