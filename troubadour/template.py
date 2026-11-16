"""Explicit ChatML for single-turn, non-thinking classification; same for both models."""
from pathlib import Path
TEMPLATE=Path(__file__).with_name('chat_template.jinja').read_text()

def configure_tokenizer(tokenizer):
    # MLX wraps a transformers tokenizer. Set the underlying template, never cached files.
    wrapped=getattr(tokenizer,'_tokenizer',None)
    target=wrapped if callable(getattr(wrapped,'apply_chat_template',None)) else tokenizer
    target.chat_template=TEMPLATE
    return tokenizer
