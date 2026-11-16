import random
from datasets import load_dataset
from troubadour.dataset_export import to_hf_dataset
from troubadour.worlds import Oracle,make_episode,rules


def test_all_splits_load_together_with_nullable_contrast_metadata(tmp_path):
    row=make_episode(rules()[7].mask,random.Random(2),Oracle(),8)
    ordinary=to_hf_dataset([row])
    contrast=to_hf_dataset([dict(row,contrast_group='same-evidence')])
    assert ordinary.features==contrast.features
    assert ordinary[0]['contrast_group'] is None
    files={}
    for name,ds in [('train',ordinary),('validation_contrast',contrast)]:
        files[name]=str(tmp_path/f'{name}.parquet');ds.to_parquet(files[name])
    loaded=load_dataset('parquet',data_files=files,cache_dir=str(tmp_path/'cache'))
    assert loaded['train'][0]['messages'][-1]['content']==row['answer']
    assert loaded['validation_contrast'][0]['contrast_group']=='same-evidence'
