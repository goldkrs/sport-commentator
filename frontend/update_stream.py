from pathlib import Path
path = Path('frontend/public/app.js')
lines = path.read_text().splitlines()
new_lines = []
inserted = False
for line in lines:
    stripped = line.strip()
    if 'playbackInProgress' in stripped:
        continue
    new_lines.append(line)
    if stripped == 'stopActiveAudio();' and not inserted:
        new_lines.append('  playbackStarted = false;')
        new_lines.append('  audioPlaybackQueue.length = 0;')
        inserted = True
path.write_text('\r\n'.join(new_lines) + '\r\n')
