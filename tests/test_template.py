from jinja2 import Environment
from troubadour.template import TEMPLATE,configure_tokenizer

def test_chatml_prefix_has_no_training_only_think_block():
    template=Environment().from_string(TEMPLATE)
    prompt=[{'role':'system','content':'Answer one letter.'},{'role':'user','content':'Query'}]
    prefix=template.render(messages=prompt,add_generation_prompt=True)
    full=template.render(messages=prompt+[{'role':'assistant','content':'A'}],add_generation_prompt=False)
    assert full==prefix+'A<|im_end|>\n'
    assert '<think>' not in full


def test_configuration_handles_raw_and_wrapped_tokenizers():
    class Raw:
        _tokenizer=object()
        chat_template='old'
        def apply_chat_template(self):pass
    class Wrapped:
        def __init__(self):self._tokenizer=Raw()
    raw=Raw();wrapped=Wrapped()
    configure_tokenizer(raw);configure_tokenizer(wrapped)
    assert raw.chat_template==TEMPLATE
    assert wrapped._tokenizer.chat_template==TEMPLATE
