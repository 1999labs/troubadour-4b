"""Small general instruction checks, not a comprehensive capability benchmark."""
import argparse
import json
from pathlib import Path
import time
from .predict import load_predictor
from .evaluate import safe_power

CASES=[
 ('addition','What is 19 + 23? Reply with only the integer.','42'),
 ('multiply','What is 7 times 8? Reply with only the integer.','56'),
 ('sort','Sort 9, 2, 5 in ascending order. Reply exactly as comma-separated integers without spaces.','2,5,9'),
 ('reverse','Reverse the letters in the word lamp. Reply only with the reversed string.','pmal'),
 ('copy','Reply with exactly this word: lantern','lantern'),
 ('extract','The traveler is named Mira and her companion is Oren. Reply only with the traveler\'s name.','Mira'),
 ('count','How many items are in this list: apple, pear, plum, peach? Reply only with the integer.','4'),
 ('case','Convert the word TRAVEL to lowercase. Reply with only the converted word.','travel'),
 ('negation','The box contains a red token and no blue tokens. Does it contain a blue token? Reply exactly yes or no.','no'),
 ('conditional','If the lamp is on, reply sun. Otherwise reply moon. The lamp is off.','moon'),
 ('format','Reply with the JSON object whose only key is x and whose value is 3. Use no spaces or markdown.','{"x":3}'),
 ('sequence','What comes next in 2, 4, 6, 8? Reply only with the integer.','10'),
]


def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter');p.add_argument('--out',required=True);a=p.parse_args()
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    model,tok=load_predictor(a.adapter);out=[]
    for name,prompt,expected in CASES:
        safe_power();start=time.time()
        text=tok.apply_chat_template([{'role':'system','content':'You are a helpful assistant. Follow the requested output format.'},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        response=generate(model,tok,prompt=text,max_tokens=64,sampler=make_sampler(temp=0),verbose=False)
        out.append({'name':name,'prompt':prompt,'expected':expected,'response':response,'correct':response.strip()==expected,'seconds':time.time()-start})
    report={'note':'12 simple instruction smoke checks, not evidence of broad general capability','passed':sum(r['correct'] for r in out),'total':len(out),'cases':out}
    Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
