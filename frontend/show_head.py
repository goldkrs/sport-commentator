from pathlib import Path
text = Path('frontend/public/app.js').read_text()
end = text.index( const startBtn)
print(repr(text[:end]))
