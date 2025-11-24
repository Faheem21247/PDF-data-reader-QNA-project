import re

def clean_text(text):
    if not text:
        return ""

    # 1. Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)

    # 2. Remove leading/trailing spaces on each line
    text = "\n".join(line.strip() for line in text.splitlines())

    # 3. Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    # 4. Replace non-breaking spaces & weird unicode
    text = text.replace("\xa0", " ")

    # 5. Remove random symbols that break models
    text = re.sub(r'[•●▪·■□▶]+', '', text)

    # 6. Remove empty lines again
    text = "\n".join([line for line in text.splitlines() if line.strip()])

    return text
