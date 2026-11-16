from troubadour.evaluate import parse_answer, summarize

def test_strict_answer_parser():
    assert parse_answer(' A\n')=='A'
    assert parse_answer('I think A')=='INVALID'
    assert parse_answer('A or B')=='INVALID'

def test_summary_does_not_reward_abstaining_everywhere():
    rows=[{'episode_id':str(i),'k':8,'expected':c,'prediction':'U'} for i,c in enumerate('ABU')]
    s=summarize(rows)
    assert s['accuracy']==1/3
    assert s['answerable_accuracy']==0
    assert s['uncertainty_recall']==1
    assert s['uncertainty_precision']==1/3
