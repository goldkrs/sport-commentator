from pathlib import Path
path = Path('frontend/public/app.js')
lines = path.read_text().splitlines()
new_lines = []
for line in lines:
    stripped = line.strip()
    if stripped == 'let playbackInProgress = false;':
        continue
    new_lines.append(line)
    if stripped == 'let suppressSegmentSeek = false;':
        new_lines.append('let playbackStarted = false;')
        new_lines.append('const audioPlaybackQueue = [];')
path.write_text('\r\n'.join(new_lines) + '\r\n')
