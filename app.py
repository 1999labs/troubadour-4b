"""Local interactive demo. No online API or paid inference required."""
import argparse
import gc
import random
import gradio as gr
from troubadour.worlds import Oracle,rules,make_episode,prompt
from troubadour.predict import load_predictor,predict

oracle=Oracle()

def visit():
    rng=random.Random()
    row=make_episode(rng.choice(rules()).mask,rng,oracle,k=8,want_known=rng.choice([True,False]))
    return row,prompt(row),''


def observe(row,k):
    if not row:return 'Create a settlement first.'
    short=dict(row);short['observations']=row['observations'][:int(k)]
    return prompt(short)


def judge(row,k,adapter):
    if not row:return 'Create a settlement first.'
    short=dict(row);short['observations']=row['observations'][:int(k)]
    model,tok=load_predictor(adapter or None)
    result=predict(model,tok,short)
    del model,tok;gc.collect()
    import mlx.core as mx
    mx.clear_cache()
    truth,count=oracle.answer(short['observations'],short['query'])
    names={'A':'Respect','B':'Breach','U':'Insufficient evidence'}
    return f"Model: **{names[result['answer']]}**\n\nChecker: **{names[truth]}**\n\n{count} allowed rules fit the visible observations."


def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',default='');p.add_argument('--port',type=int,default=7860);a=p.parse_args()
    def new_visit():
        row,view,_=visit()
        return row,view,'','',8
    def change_evidence(row,k):
        return observe(row,k),'',''
    def compare_models(row,k):
        original=judge(row,k,'')
        trained=judge(row,k,a.adapter) if a.adapter else 'Start the app with --adapter to compare Troubadour.'
        return original,trained
    with gr.Blocks(title='Troubadour') as demo:
        gr.Markdown('# Troubadour\nTest street smarts in an unfamiliar fictional settlement. A = respect, B = breach, U = insufficient evidence. These are fictional conventions, not descriptions of real cultures.')
        gr.Markdown('Reveal more observations, then compare the models. Random demo worlds may overlap training rules; this interface is not the held-out benchmark.')
        state=gr.State();new=gr.Button('Visit a new settlement');k=gr.Slider(0,8,value=8,step=1,label='Visible observations')
        view=gr.Textbox(label='Encounter',lines=16);run=gr.Button('Compare models')
        with gr.Row():
            with gr.Column():
                gr.Markdown('### Original Qwen');base_result=gr.Markdown()
            with gr.Column():
                gr.Markdown('### Troubadour');trained_result=gr.Markdown()
        new.click(new_visit,outputs=[state,view,base_result,trained_result,k])
        k.change(change_evidence,inputs=[state,k],outputs=[view,base_result,trained_result])
        run.click(compare_models,inputs=[state,k],outputs=[base_result,trained_result])
    demo.launch(server_name='127.0.0.1',server_port=a.port)

if __name__=='__main__':main()
