"""Keep each six-example balanced group inside one optimizer update."""
import random


def group_indices(n,seed):
    if n<6 or n%6:raise ValueError('Training requires complete groups of six')
    rng=random.Random(seed)
    while True:
        groups=list(range(n//6));rng.shuffle(groups)
        for group in groups:yield tuple(range(group*6,group*6+6))


def balanced_indices(labels, seed):
    """Reorder an existing balanced dataset into six-example updates, without new data."""
    pools={label:[i for i,value in enumerate(labels) if value==label] for label in sorted(set(labels))}
    sizes={len(values) for values in pools.values()}
    if len(pools)!=3 or len(sizes)!=1 or not sizes or next(iter(sizes))%2:
        raise ValueError('Need three equally frequent labels, with an even count per label')
    rng=random.Random(seed)
    while True:
        for values in pools.values():rng.shuffle(values)
        groups=[tuple(pools[label][offset+j] for j in range(2) for label in pools) for offset in range(0,next(iter(sizes)),2)]
        rng.shuffle(groups)
        yield from groups
