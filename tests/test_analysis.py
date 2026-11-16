from troubadour.analyze import compare

def test_cluster_bootstrap_is_paired():
    base=[];candidate=[]
    for i in range(4):
        r={'episode_id':str(i),'k':8,'style':'table','society_pair':i//2,'expected':'A','prediction':'B'}
        base.append(r);candidate.append(r|{'prediction':'A'})
    report=compare(base,candidate)
    assert report['cluster_mean_gain']==1
    assert report['cluster_bootstrap_95ci']==[1,1]
