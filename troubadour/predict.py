"""Use the local trained adapter with the same inference path as the benchmark."""
import argparse
import json
from pathlib import Path
from .choice import choice_probabilities
from .evaluate import BASE,BASE_PATH,REVISION
from .template import configure_tokenizer
from .worlds import messages


def load_predictor(adapter=None):
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    base=BASE_PATH if BASE_PATH.exists() else snapshot_download(BASE,revision=REVISION)
    model,tok=load(str(base),adapter_path=adapter)
    configure_tokenizer(tok)
    return model,tok


def predict(model,tok,episode,style='table'):
    # Only observations and query are read by messages(completion=False).
    ids=tok.apply_chat_template(messages(episode,style=style,completion=False),tokenize=True,return_dict=False,add_generation_prompt=True)
    labels={c:tok.encode(c,add_special_tokens=False)[0] for c in ('A','B','U')}
    scores=dict(zip(('A','B','U'),choice_probabilities(model,[ids],labels)[0]))
    return {'answer':max(scores,key=scores.get),'choice_probabilities':scores}


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--adapter');p.add_argument('--model-repo');a=p.parse_args()
    adapter=a.adapter
    if a.model_repo:
        from huggingface_hub import snapshot_download
        adapter=str(Path(snapshot_download(a.model_repo))/'adapter')
    model,tok=load_predictor(adapter)
    episode=json.loads(Path(a.episode).read_text())
    print(json.dumps(predict(model,tok,episode),indent=2))

if __name__=='__main__':main()
