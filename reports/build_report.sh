#!/bin/sh
# Build reports/report.pdf from reports/report.md: pandoc -> self-contained HTML -> headless Chrome.
# Needs pandoc and Google Chrome (macOS path below; override with CHROME=...).
# Chrome writes the PDF but may not exit afterwards, so we wait for the file and stop it ourselves.
set -e
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PROFILE="$(mktemp -d)"
rm -f report.pdf

pandoc report.md --standalone --embed-resources --number-sections --css report.css \
    --metadata pagetitle="Explaining customer churn" -o report.html
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --user-data-dir="$PROFILE" \
    --print-to-pdf=report.pdf report.html >/dev/null 2>&1 &
CHROME_PID=$!

for _ in $(seq 1 60); do
    [ -s report.pdf ] && break
    sleep 1
done
sleep 2                                  # let Chrome finish writing
kill "$CHROME_PID" 2>/dev/null || true
rm -f report.html
chmod -R u+rwx "$PROFILE" 2>/dev/null || true   # Chrome leaves some folders read-only
rm -rf "$PROFILE" 2>/dev/null || true   # macOS cleans /var/folders anyway
[ -s report.pdf ] || { echo "report.pdf was not produced" >&2; exit 1; }
echo "reports/report.pdf written"
