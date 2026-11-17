"""Input-only controls: nearby examples, query-blind voting, and the public solver."""
import argparse
import json
from pathlib import Path
from .worlds import Oracle
from .evaluate import summarize

KINDS = ('nearest_example', 'majority_label', 'always_U', 'reference_solver')


def predict(kind, observations, query):
    if kind == 'always_U':
        return 'U'
    if kind == 'reference_solver':
        return Oracle().answer(observations, query)[0]
    if not observations:
        return 'U'
    if kind == 'nearest_example':
        distance = min((state ^ query).bit_count() for state, _ in observations)
        labels = {value for state, value in observations if (state ^ query).bit_count() == distance}
        return 'U' if len(labels) != 1 else ('A' if next(iter(labels)) else 'B')
    if kind == 'majority_label':
        votes = sum(value for _, value in observations)
        return 'U' if 2 * votes == len(observations) else ('A' if 2 * votes > len(observations) else 'B')
    raise ValueError('Unknown baseline: ' + kind)


def score_rows(rows, kind):
    out = []
    for row in rows:
        prediction = predict(kind, row['observations'], row['query'])
        result = {'episode_id': row['id'], 'society_pair': row['society_pair'],
                  'family': row['family'], 'k': row['k'], 'style': 'table',
                  'expected': row['answer'], 'prediction': prediction}
        if row.get('contrast_group'):
            result['contrast_group'] = row['contrast_group']
        out.append(result)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    result = {kind: summarize(score_rows(rows, kind)) for kind in KINDS}
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
