import random
from troubadour.worlds import *


def test_complement_closure_and_unique_rules():
    masks=[r.mask for r in rules()]
    assert len(masks)==len(set(masks))
    assert all(FULL^m in masks for m in masks)


def test_oracle_agrees_with_independent_reference_and_hidden_world():
    oracle=Oracle();rng=random.Random(43)
    for _ in range(400):
        rule=rng.choice(rules());k=rng.choice([0,1,2,4,8,12,20,31])
        row=make_episode(rule.mask,rng,oracle,k)
        assert row['answer']==independent_answer(row['observations'],row['query'])
        if row['answer']!='U':assert row['answer']==('A' if (rule.mask>>row['query'])&1 else 'B')
        reverse=complement(row,oracle)
        assert reverse['answer']=={'A':'B','B':'A','U':'U'}[row['answer']]
        assert row['query'] not in [s for s,_ in row['observations']]


def test_empty_evidence_is_unknown():
    oracle=Oracle()
    assert all(oracle.answer([],s)[0]=='U' for s in range(32))


def test_splits_have_no_rule_or_complement_leakage():
    split=split_manifest();seen=set()
    for key in ['train','dev','final','dev_transfer','final_transfer']:
        expanded={m for p in split[key] for m in (p,FULL^p)}
        assert not seen.intersection(expanded);seen.update(expanded)
    assert len(seen)==len(rules())
    train_supports={support(m) for m in split['train']}
    assert all(support(m) not in train_supports for m in split['final_transfer'])


def test_public_prompt_does_not_reveal_private_metadata():
    row=make_episode(rules()[37].mask,random.Random(7),Oracle(),8)
    s=prompt(row)
    for name in ['rule_mask','society_pair','consistent_rules','family']:
        assert name not in s
    assert str(row['rule_mask']) not in s
