from pathlib import Path
for i, line in enumerate(Path('frontend/public/app.js').read_text().splitlines(), start=1):
    if i<=20:
        print(i, repr(line))
