from troubadour.calibrate import adjusted,fit

def test_uncertainty_adjustment_preserves_a_b_symmetry():
    rows=[{'episode_id':'a','k':8,'expected':'A','prediction':'U','label_probs':{'A':.2,'B':.1,'U':.7}},
          {'episode_id':'b','k':8,'expected':'B','prediction':'U','label_probs':{'A':.1,'B':.2,'U':.7}}]
    result=adjusted(rows,-3)
    assert [r['prediction'] for r in result]==['A','B']
    assert fit(rows)['development_accuracy']==1
