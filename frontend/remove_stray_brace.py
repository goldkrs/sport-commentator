from pathlib import Path
path = Path('frontend/public/app.js')
text = path.read_text()
old = '}\r\n\r\n\r\nfunction enqueueSegment'
if old not in text:
    raise SystemExit('pattern not found')
text = text.replace(old, '\r\n\r\nfunction enqueueSegment', 1)
path.write_text(text)
