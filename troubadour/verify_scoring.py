"""GPU parity test: optimized scoring must match original full-logit scalar scoring."""
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from .evaluate import BASE_PATH
from .worlds import messages
from .choice import choice_probabilities

model,tok=load(str(BASE_PATH))
from .template import configure_tokenizer
configure_tokenizer(tok)
rows=[json.loads(l) for l in Path('data/dev.jsonl').read_text().splitlines()][:4]
inputs=[tok.apply_chat_template(messages(r,completion=False),tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
labels={c:tok.encode(c,add_special_tokens=False)[0] for c in ('A','B','U')}
expected=[]
for tokens in inputs:
 logits=model(mx.array([tokens]))[0,-1]
 p=mx.softmax(logits[mx.array(list(labels.values()))]);mx.eval(p);expected.append(p.tolist())
actual=choice_probabilities(model,inputs,labels)
errors=[abs(a-b) for v,w in zip(expected,actual) for a,b in zip(v,w)]
assert max(errors)<0.01,(actual,expected)
assert [max(range(3),key=v.__getitem__) for v in expected]==[max(range(3),key=v.__getitem__) for v in actual]
report={'max_probability_difference':max(errors),'same_predictions':True,'examples':4,'expected':expected,'batched':actual}
Path('artifacts/scoring_parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
