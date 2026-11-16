"""One explicit Arrow schema shared by every training and evaluation split."""
from .worlds import messages


def to_hf_dataset(rows):
    from datasets import Dataset,Features,List,Value
    integer=Value('int64');string=Value('string')
    features=Features({
        'rule_mask':integer,'observations':List(List(integer)),
        'query':integer,'answer':string,'consistent_rules':integer,'k':integer,
        'id':string,'society_pair':integer,'family':string,'contrast_group':string,
        'messages':List({'role':string,'content':string}),
    })
    prepared=[dict(row,contrast_group=row.get('contrast_group'),messages=messages(row)) for row in rows]
    return Dataset.from_list(prepared,features=features)
