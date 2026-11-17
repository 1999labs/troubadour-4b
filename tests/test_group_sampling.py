import pytest
from troubadour.group_sampling import group_indices


def test_balanced_groups_are_not_interleaved_and_every_example_is_used():
    batches=group_indices(60,17);epoch=[next(batches) for _ in range(10)]
    assert sorted(i for group in epoch for i in group)==list(range(60))
    for group in epoch:
        assert len({i//6 for i in group})==1
        assert group==tuple(range(group[0],group[0]+6))
    repeated=group_indices(60,17)
    assert epoch==[next(repeated) for _ in range(10)]
    with pytest.raises(ValueError):next(group_indices(59,17))


def test_balanced_sampler_preserves_an_epoch_and_balances_every_update():
    from collections import Counter
    from itertools import islice
    from troubadour.group_sampling import balanced_indices
    labels=list('ABUUBAABUUBA')
    groups=list(islice(balanced_indices(labels,17),2))
    assert sorted(i for group in groups for i in group)==list(range(len(labels)))
    assert all(Counter(labels[i] for i in group)==Counter({'A':2,'B':2,'U':2}) for group in groups)
    assert groups==list(islice(balanced_indices(labels,17),2))
