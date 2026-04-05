from pathlib import Path
lines = Path('frontend/public/app.js').read_text().splitlines()
for idx, line in enumerate(lines):
    if 'function flushPendingTexts' in line:
        start = idx
        break
else:
    raise SystemExit('flush function not found')
for i in range(start, start+120):
    print(i+1, repr(lines[i]))
