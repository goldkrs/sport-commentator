from pathlib import Path
lines = Path('frontend/public/app.js').read_text().splitlines()
for idx in range(840, 880):
    print(idx + 1, lines[idx])
