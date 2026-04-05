from pathlib import Path
lines = Path('frontend/public/app.js').read_text().splitlines()
for idx in range(460, 540):
    print(idx + 1, repr(lines[idx]))
