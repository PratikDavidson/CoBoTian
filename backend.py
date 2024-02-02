import torch
import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


torch.set_default_device("cuda")

@st.cache_resource
def init_model():
    model = Model()
    return model

def process_text(text):
    text = text[text.find('Output:'):text.find('"""')]
    text = text[text.find('"')+1:text.rfind('"')]
    return text

def stream_data(prompt=""):
    for word in prompt.split():
        yield word + " "
        time.sleep(0.02)

class Model:

    def __init__(self):
        checkpoint = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", trust_remote_code=True)      

    def process(self, prompt):

        inputs = self.tokenizer(f'''Instruct: Create a funny joke on the given phrase "{prompt}".
                                    Output:
                                 ''', return_tensors='pt', return_attention_mask=True)
        inputs = inputs.to('cuda')
        outputs = self.model.generate(**inputs, max_length=200, pad_token_id=self.tokenizer.eos_token_id)
        text = process_text(self.tokenizer.batch_decode(outputs)[0])
        return text