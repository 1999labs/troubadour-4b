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


def test_complement_pair_metric_separates_trivial_unknown_pairs():
    rows=[]
    for name,labels in [('unknown','UU'),('known','AB')]:
        for i,c in enumerate(labels):
            rows.append({'episode_id':name+('-reverse' if i else ''),'k':8,'expected':c,'prediction':'U'})
    result=summarize(rows)
    assert result['paired_both_correct']==0.5
    assert result['paired_answerable_both_correct']==0.0


def test_uncertainty_ranking_separates_ordering_from_response_bias():
    from troubadour.evaluate import uncertainty_auc
    from math import exp
    rows=[{'expected':c,'label_probs':p} for c,p in [
        ('U',{'A':.1,'B':.1,'U':.8}),
        ('A',{'A':.7,'B':.2,'U':.1}),
        ('B',{'A':.1,'B':.8,'U':.1})]]
    assert uncertainty_auc(rows)==1.0
    shifted=[]
    for row in rows:
        p=dict(row['label_probs']);p['U']*=exp(-2);total=sum(p.values())
        shifted.append(dict(row,label_probs={k:v/total for k,v in p.items()}))
    assert uncertainty_auc(shifted)==1.0
    tied=[dict(row,label_probs={'A':1/3,'B':1/3,'U':1/3}) for row in rows]
    assert uncertainty_auc(tied)==.5
