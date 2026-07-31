"""Fix double quotes in translate_v_story.py"""

import re

with open('manual/translate_v_story.py', 'r', encoding='utf-8') as f:
    content = f.read()


def fix_quotes(m):
    prefix = m.group(1)
    value = m.group(2)
    value = value.replace("'", "\\'")
    return f"{prefix}'{value}',"


content = re.sub(
    r'^(\s+\d+:\s+)"(.*)",\s*$',
    fix_quotes,
    content,
    flags=re.MULTILINE
)

with open('manual/translate_v_story.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed!")
