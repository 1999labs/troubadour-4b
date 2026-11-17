"""Explicit ChatML for single-turn, non-thinking classification; same for both models."""
from pathlib import Path
TEMPLATE=Path(__file__).with_name('chat_template.jinja').read_text()

def configure_tokenizer(tokenizer):
    # MLX wraps a transformers tokenizer. Set the underlying template, never cached files.
    wrapped=getattr(tokenizer,'_tokenizer',None)
    target=wrapped if callable(getattr(wrapped,'apply_chat_template',None)) else tokenizer
    target.chat_template=TEMPLATE
    return tokenizer


def renderer_sha256():
    """Bind evaluation output to the exact renderer and chat-template source."""
    import hashlib
    root=Path(__file__).parent
    payload=b'\0'.join((root/name).read_bytes() for name in ['worlds.py','template.py','chat_template.jinja'])
    return hashlib.sha256(payload).hexdigest()
