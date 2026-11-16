from troubadour.learning_curve import curve

def test_zero_evidence_does_not_count_as_resolving_a_convention():
    rows=[{'episode_id':'x','k':0,'expected':'U','prediction':'U'}, {'episode_id':'x','k':8,'expected':'A','prediction':'A'}]
    c=curve(rows)
    assert c[0]['joint_accuracy_including_uncertainty']==1
    assert c[0]['justified_correct_resolution']==0
    assert c[1]['justified_correct_resolution']==1
