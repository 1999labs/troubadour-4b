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
