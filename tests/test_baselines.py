from troubadour.baselines import predict


def test_nearest_requires_agreement_even_if_most_neighbors_vote_a():
    observations = [(0, 1), (3, 1), (5, 0)]
    assert predict('nearest_example', observations, 1) == 'U'
    assert predict('majority_label', observations, 1) == 'A'


def test_nearest_and_majority_have_distinct_information():
    observations = [(0, 1), (7, 0)]
    assert predict('nearest_example', observations, 1) == 'A'
    assert predict('majority_label', observations, 1) == 'U'
    assert predict('nearest_example', [], 1) == 'U'
    assert predict('reference_solver', [], 1) == 'U'
