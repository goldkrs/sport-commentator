from pathlib import Path
path = Path('frontend/public/app.js')
lines = path.read_text().splitlines()
new_lines = []
blank = 0
for line in lines:
    stripped = line.rstrip()
    if stripped == '':
        blank += 1
        if blank > 1:
            continue
        new_lines.append('')
    else:
        blank = 0
        new_lines.append(stripped)
path.write_text('\r\n'.join(new_lines) + '\r\n')
