"""Finite social-convention worlds; deterministic generation, no teacher API."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
import hashlib
import json
import random
from pathlib import Path

N_CUES = 5
N_STATES = 1 << N_CUES
FULL = (1 << N_STATES) - 1
CUE_NAMES = ('rank', 'setting', 'gesture', 'favor', 'origin')
VALUES = (('lower', 'higher'), ('private', 'public'), ('accept', 'decline'), ('no', 'yes'), ('local', 'visitor'))
GRAMMAR = ('Possible customs: constant A or B; one cue; AND of two or three cue tests; '
           'majority of three cue tests; odd/even parity of two or three cues; '
           'or one cue selecting which of two other cues decides. Any cue test or output can be reversed. '
           'No other rules are possible. The actual rule is not given.')
SYSTEM = ('You are a traveler interpreting a fictional settlement. Infer its convention ONLY from the observations. '
          'A means respect; B means a breach of etiquette. These meanings can reverse between settlements. '
          'Return A or B only if EVERY allowed rule consistent with the observations agrees on the query. '
          'Otherwise return U (insufficient evidence). Reply with exactly one letter: A, B, or U.')


def bits(state):
    return tuple((state >> i) & 1 for i in range(N_CUES))


def support(mask):
    return tuple(i for i in range(N_CUES) if any(((mask >> s) & 1) != ((mask >> (s ^ (1 << i))) & 1) for s in range(N_STATES)))


@dataclass(frozen=True)
class Rule:
    mask: int
    family: str
    description: str


@lru_cache(maxsize=1)
def rules():
    found = {}
    def add(fn, family, desc):
        mask = sum(int(bool(fn(bits(s)))) << s for s in range(N_STATES))
        if mask not in found:
            found[mask] = Rule(mask, family, desc)
    def test_desc(i, n):
        return f'{CUE_NAMES[i]}={VALUES[i][1-n]}'
    for v in (0, 1):
        add(lambda x, v=v: v, 'constant', f'Always {"A" if v else "B"}')
    for width in (1, 2, 3):
        for inds in combinations(range(N_CUES), width):
            for neg in product((0, 1), repeat=width):
                desc = ' AND '.join(test_desc(i, n) for i, n in zip(inds, neg))
                for flip in (0, 1):
                    add(lambda x, inds=inds, neg=neg, flip=flip: int(all(x[i] ^ n for i, n in zip(inds, neg))) ^ flip,
                        'single' if width == 1 else f'and{width}', f'{"B" if flip else "A"} exactly when {desc}; otherwise {"A" if flip else "B"}')
            if width > 1:
                for flip in (0, 1):
                    add(lambda x, inds=inds, flip=flip: (sum(x[i] for i in inds) % 2) ^ flip,
                        f'parity{width}', f'{"B" if flip else "A"} when an odd number of these tests hold: ' + ', '.join(test_desc(i,0) for i in inds) + f'; otherwise {"A" if flip else "B"}')
            if width == 3:
                for neg in product((0, 1), repeat=3):
                    add(lambda x, inds=inds, neg=neg: sum(x[i] ^ n for i, n in zip(inds, neg)) >= 2,
                        'majority3', 'A when at least two of ' + ', '.join(test_desc(i, n) for i, n in zip(inds, neg)) + '; otherwise B')
                for selector in inds:
                    rest = [i for i in inds if i != selector]
                    for a, b in (rest, rest[::-1]):
                        for na, nb in product((0, 1), repeat=2):
                            add(lambda x, selector=selector, a=a, b=b, na=na, nb=nb: (x[a] ^ na) if x[selector] else (x[b] ^ nb),
                                'switch3', f'If {CUE_NAMES[selector]}={VALUES[selector][1]}, A iff {test_desc(a,na)}; otherwise A iff {test_desc(b,nb)}; B otherwise')
    return tuple(found[m] for m in sorted(found))


class Oracle:
    def __init__(self):
        self.rules = rules()
        self.all = (1 << len(self.rules)) - 1
        self.by_state = []
        for s in range(N_STATES):
            one = sum(1 << i for i, r in enumerate(self.rules) if (r.mask >> s) & 1)
            self.by_state.append((self.all ^ one, one))

    def survivors(self, observations):
        alive = self.all
        for state, value in observations:
            alive &= self.by_state[state][value]
        if not alive:
            raise ValueError('Observations contradict the declared grammar')
        return alive

    def answer(self, observations, query):
        alive = self.survivors(observations)
        n0 = (alive & self.by_state[query][0]).bit_count()
        n1 = (alive & self.by_state[query][1]).bit_count()
        return ('U' if n0 and n1 else 'A' if n1 else 'B'), n0 + n1


def independent_answer(observations, query):
    """Slow reference: direct rule filtering independent of bitset implementation."""
    survivors = [r for r in rules() if all(((r.mask >> s) & 1) == v for s, v in observations)]
    if not survivors:
        raise ValueError('No consistent rule')
    values = {(r.mask >> query) & 1 for r in survivors}
    return 'U' if len(values) > 1 else 'A' if 1 in values else 'B'


def split_manifest(seed=73193):
    rng = random.Random(seed)
    masks = {r.mask for r in rules()}
    assert all(FULL ^ m in masks for m in masks)
    pairs = sorted(m for m in masks if m < (FULL ^ m))
    ood_supports = {(0, 1, 2), (2, 3, 4)}
    iid = [m for m in pairs if support(m) not in ood_supports]
    ood = [m for m in pairs if support(m) in ood_supports]
    rng.shuffle(iid); rng.shuffle(ood)
    nd = max(12, int(len(iid) * .16)); nt = max(20, int(len(iid) * .2))
    result = {'seed':seed, 'grammar_size':len(masks), 'train':iid[nd+nt:], 'dev':iid[:nd], 'final':iid[nd:nd+nt], 'dev_transfer':ood[:len(ood)//2], 'final_transfer':ood[len(ood)//2:]}
    return result


def make_episode(mask, rng, oracle, k=8, want_known=None):
    for attempt in range(1000):
        states = rng.sample(range(N_STATES), k + 1)
        observations = [(s, (mask >> s) & 1) for s in states[:k]]
        query = states[-1]
        answer, n = oracle.answer(observations, query)
        if want_known is None or (answer != 'U') == want_known:
            break
    else:
        raise RuntimeError('Unable to sample requested evidence condition')
    payload = {'rule_mask':mask, 'observations':observations, 'query':query, 'answer':answer, 'consistent_rules':n, 'k':k}
    payload['id'] = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:20]
    payload['society_pair'] = min(mask,FULL^mask)
    payload['family'] = next(r.family for r in rules() if r.mask == mask)
    return payload


def complement(row, oracle):
    result = dict(row)
    result['rule_mask'] = FULL ^ row['rule_mask']
    result['observations'] = [(s,1-v) for s,v in row['observations']]
    result['answer'],result['consistent_rules'] = oracle.answer(result['observations'],result['query'])
    result['id'] = row['id'] + '-reverse'
    return result


def render_state(s, style='table'):
    x = bits(s)
    v = [VALUES[i][x[i]] for i in range(N_CUES)]
    if style == 'symbols':
        return ' '.join(map(str,x))
    if style == 'prose':
        return f'The recipient is {v[0]} in rank, the exchange is {v[1]}, they {v[2]} the token, a prior favor is {"present" if x[3] else "absent"}, and the recipient is a {v[4]}.'
    return ' | '.join(v)


def prompt(row, style='table', explicit_rule=False):
    legend = ('Columns: recipient rank relative to giver | setting | token gesture | prior favor | recipient origin.\n'
              'Each row is an independent encounter in the SAME settlement.\n')
    if style == 'symbols':
        legend = 'Each row gives five binary cues x0 x1 x2 x3 x4. All rows use the SAME hidden rule.\n'
    if style == 'prose':
        legend = 'Each observation is an independent encounter in the SAME settlement.\n'
    lines = [GRAMMAR,legend,'Observations:']
    for i,(s,v) in enumerate(row['observations'],1):
        lines.append(f'{i}. {render_state(s,style)} => {"A" if v else "B"}')
    if not row['observations']:
        lines.append('(none)')
    lines.extend(['Query: ' + render_state(row['query'],style),'Answer A, B, or U:'])
    if explicit_rule:
        rule = next(r for r in rules() if r.mask == row['rule_mask'])
        lines.insert(1,'For this exercise the local rule is given: '+rule.description+'. Use this rule directly; do not answer U.')
    return '\n'.join(lines)


def messages(row, style='table', explicit_rule=False, completion=True):
    system = 'Apply the explicitly supplied fictional convention to the query. A means respect, B means breach. Reply with exactly A or B.' if explicit_rule else SYSTEM
    result=[{'role':'system','content':system},{'role':'user','content':prompt(row,style,explicit_rule)}]
    if completion:
        answer = ('A' if (row['rule_mask'] >> row['query']) & 1 else 'B') if explicit_rule else row['answer']
        result.append({'role':'assistant','content':answer})
    return result


def write_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
