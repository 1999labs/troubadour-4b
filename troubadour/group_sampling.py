"""Keep each six-example balanced group inside one optimizer update."""
import random


def group_indices(n,seed):
    if n<6 or n%6:raise ValueError('Training requires complete groups of six')
    rng=random.Random(seed)
    while True:
        groups=list(range(n//6));rng.shuffle(groups)
        for group in groups:yield tuple(range(group*6,group*6+6))
