from pathlib import Path
path = Path('frontend/public/app.js')
text = path.read_text()
old = '  form.append( prompt, COMMENTARY_INSTRUCTION);'
if old not in text:
    raise SystemExit('old text missing')
new = old + '\n  form.append(max_new_tokens, 30);'
text = text.replace(old, new, 1)
path.write_text(text)
