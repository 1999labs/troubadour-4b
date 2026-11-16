"""MLX SFT with a verified, identical non-thinking ChatML template."""
import os
# MLX maintainer-recommended, process-local workaround for macOS display contention.
os.environ.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")
from mlx_lm import lora
import mlx.core as mx
from .template import configure_tokenizer
from .evaluate import BASE,BASE_PATH,REVISION
_original_load=lora.load

def load_non_thinking(*args,**kwargs):
    if args and args[0]==BASE:
        from huggingface_hub import snapshot_download
        path=str(BASE_PATH) if BASE_PATH.exists() else snapshot_download(BASE,revision=REVISION)
        args=(path,)+args[1:]
    model,tokenizer=_original_load(*args,**kwargs)
    return model,configure_tokenizer(tokenizer)

if __name__=='__main__':
    mx.set_cache_limit(256 * 1024 * 1024)
    mx.set_memory_limit(10 * 1024 * 1024 * 1024)
    lora.load=load_non_thinking
    lora.main()
