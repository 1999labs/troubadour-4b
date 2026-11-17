import random
import pytest
from troubadour.audit import audit_rows
from troubadour.worlds import Oracle, make_episode, rules


def episode():
    return make_episode(rules()[37].mask,random.Random(7),Oracle(),8)


def test_audit_rejects_wrong_observations_and_answer_metadata():
    row=episode();allowed={row['rule_mask']}
    audit_rows([row],allowed)
    with pytest.raises(AssertionError):
        audit_rows([dict(row,consistent_rules=row['consistent_rules']+1)],allowed)
    obs=list(row['observations']);s,v=obs[0];obs[0]=(s,1-v)
    with pytest.raises(AssertionError):audit_rows([dict(row,observations=obs)],allowed)


def test_audit_rejects_duplicates_and_wrong_split():
    row=episode()
    with pytest.raises(AssertionError):audit_rows([row,row],{row['rule_mask']})
    with pytest.raises(AssertionError):audit_rows([row],set())
