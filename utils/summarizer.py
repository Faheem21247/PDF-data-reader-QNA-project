from transformers import pipeline

# Load the summarizer model once (better performance)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text(text, max_chars=2000):
    """
    Break long text into smaller chunks because summarizer has token limits.
    """
    chunks = []
    while len(text) > max_chars:
        # find nearest period to split cleanly
        split_index = text.rfind('.', 0, max_chars)
        if split_index == -1:
            split_index = max_chars
        chunks.append(text[:split_index + 1])
        text = text[split_index + 1:]
    chunks.append(text)
    return chunks

def generate_summary(text):
    chunks = chunk_text(text)

    partial_summaries = []
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=150,
            min_length=40,
            do_sample=False
        )[0]['summary_text']
        partial_summaries.append(summary)

    # If multiple partial summaries → summarize them again (hierarchical)
    final_summary_text = " ".join(partial_summaries)

    if len(partial_summaries) > 1:
        final_summary = summarizer(
            final_summary_text,
            max_length=180,
            min_length=50,
            do_sample=False
        )[0]['summary_text']
        return final_summary

    return partial_summaries[0]
