from collections import defaultdict
from troubadour.build_contrast import build
from troubadour.worlds import split_manifest,independent_answer

def test_contrast_pairs_force_query_dependence():
    rows,_=build(split_manifest()['dev'][:5],123)
    grouped=defaultdict(list)
    for r in rows:
        grouped[r['contrast_group']].append(r)
        assert independent_answer(r['observations'],r['query'])==r['answer']
    assert grouped
    for pair in grouped.values():
        assert len(pair)==2
        assert pair[0]['observations']==pair[1]['observations']
        assert pair[0]['query']!=pair[1]['query']
        assert {r['answer'] for r in pair}=={'A','B'}
