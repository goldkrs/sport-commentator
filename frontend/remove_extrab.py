from pathlib import Path
lines = Path('frontend/public/app.js').read_text().splitlines()
skip_idx = None
for idx, line in enumerate(lines):
    if line == '}' and skip_idx is None:
        # look ahead for next non-empty line
        j = idx + 1
        while j < len(lines) and lines[j] == '':
            j += 1
        if j < len(lines) and lines[j].startswith('function enqueueSegment'):
            skip_idx = idx
            break
if skip_idx is None:
    raise SystemExit('extra brace not found')
new_lines = [line for i, line in enumerate(lines) if i != skip_idx]
Path('frontend/public/app.js').write_text('\r\n'.join(new_lines) + '\r\n')
