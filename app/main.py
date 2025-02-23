# Import required libraries
import numpy as np
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash
import math
import re
from   random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pickle.load(open('./model/h_data.pkl', 'rb'))
word2id = data['word2id']
max_len = data['max_len']
max_mask = data['max_mask']
vocab_size = data['vocab_size']

def clean_sentences(example):
    example["premise"] = example["premise"].lower()
    example["hypothesis"] = example["hypothesis"].lower()
    example["premise"] = re.sub("[.,!?\\-]", '',example["premise"])
    example["hypothesis"] = re.sub("[.,!?\\-]", '',example["hypothesis"])
    return example

def tokenize(example):
        output = {}
        output['input_ids'] = []
        output['att_mask'] = []
        input_ids = [word2id.get(word, word2id['[UNK]']) for word in example.split()]
        n_pad = max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        att_mask = [1 if idx != 0 else 0 for idx in input_ids]  # Create attention mask
        output['input_ids'].append(torch.tensor(input_ids))  # Convert to tensor
        output['att_mask'].append(torch.tensor(att_mask))  # Convert to tensor
        return output


n_layers = 12    # number of Encoder of Encoder Layer
n_heads  = 12    # number of heads in Multi-Head Attention
d_model  = 768  # Embedding Size
d_ff = d_model * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2

num_epoch = 10
s_model = model.BERT(
    n_layers, 
    n_heads, 
    d_model, 
    d_ff, 
    d_k, 
    n_segments, 
    vocab_size, 
    max_len, 
    device
).to(device) 

def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

classifier_head = torch.nn.Linear(768*3, 3).to(device)

s_model.load_state_dict(torch.load('./model/sbert_model.pth',map_location=torch.device('cpu')))

# Create segment_ids tensor with shape (batch_size, max_len)
segment_ids = torch.tensor([0] * max_len).unsqueeze(0).repeat(1, 1).to(device)

# Create masked_pos tensor with shape (batch_size, max_mask)
masked_pos = torch.tensor([0] * max_mask).unsqueeze(0).repeat(1, 1).to(device)

def infer(model, tokenizer, sentence_a, sentence_b, device):
    model.eval()
    sentence_a = sentence_a.lower()
    sentence_b = sentence_b.lower()
    sentence_a = re.sub("[.,!?\\-]", '',sentence_a)
    sentence_b = re.sub("[.,!?\\-]", '',sentence_b)
    inputs_a = tokenizer(sentence_a)
    # Tokenize the hypothesis
    inputs_b = tokenizer(sentence_b)
    
     # Move input IDs and attention masks to the active device
    inputs_ids_a = inputs_a['input_ids'][0].unsqueeze(0).to(device)
    attention_a = inputs_a['att_mask'][0].unsqueeze(0).to(device)
    inputs_ids_b = inputs_b['input_ids'][0].unsqueeze(0).to(device)
    attention_b = inputs_b['att_mask'][0].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # extract token embeddings from BERT at last_hidden_state
        u = model.get_last_hidden_state(inputs_ids_a, segment_ids)
        v = model.get_last_hidden_state(inputs_ids_b, segment_ids)

    u_last_hidden_state = u # all token embeddings A = batch_size, seq_len, hidden_dim
    v_last_hidden_state = v # all token embeddings B = batch_size, seq_len, hidden_dim

     # get the mean pooled vectors
    u_mean_pool = mean_pool(u_last_hidden_state, attention_a) # batch_size, hidden_dim
    v_mean_pool = mean_pool(v_last_hidden_state, attention_b) # batch_size, hidden_dim
    
    # build the |u-v| tensor
    uv = torch.sub(u_mean_pool, v_mean_pool)   # batch_size,hidden_dim
    uv_abs = torch.abs(uv) # batch_size,hidden_dim
    
    # concatenate u, v, |u-v|
    x = torch.cat([u_mean_pool, v_mean_pool, uv_abs], dim=-1) # batch_size, 3*hidden_dim
    
    # process concatenated tensor through classifier_head
    y = classifier_head(x) #batch_size, classifer

    return torch.argmax(nn.functional.softmax(y,1)).item()

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


# Create a dash application
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    dcc.Tabs(
        [
            dcc.Tab(
                label="A1 - S-BERT",
                children=[
                    html.H2(
                        children="Text similarity",
                        style={
                            "textAlign": "center",
                            "color": "#503D36",
                            "font-size": 40,
                        },
                    ),
                    dbc.Stack(
                        dcc.Textarea(
                            id="note",
                            value="""Note : This is a text similarity search engine.""",
                            style={
                                "width": "100%",
                                "height": 20,
                                "whiteSpace": "pre-line",
                                "textAlign": "center",
                            },
                            readOnly=False,
                        )
                    ),                    
                    dbc.Stack(
                        dcc.Textarea(
                            id="input1",
                            value="""Type the first sentence""",
                            style={
                                "width": "100%",
                                "height": 80,
                                "whiteSpace": "pre-line",
                            },
                            readOnly=False,
                        )
                    ),
                     dbc.Stack(
                        dcc.Textarea(
                            id="input2",
                            value="""Type the second sentence""",
                            style={
                                "width": "100%",
                                "height": 80,
                                "whiteSpace": "pre-line",
                            },
                            readOnly=False,
                        )
                    ),
                    html.Div(
                        html.Button(
                            "Check",
                            id="check",
                            n_clicks=0,
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": "100%",
                                "height": 50,
                                "background-color": "white",
                                "color": "black",
                            },
                        ),
                    ),
                    html.Br(),
                    dcc.Textarea(
                        id="result",
                        value="see here",
                        style={
                            "width": "100%",
                            "height": 100,
                            "whiteSpace": "pre-line",
                            "font-size": "1.5em",
                            "textAlign": "center",
                            "color": "#503D36",
                        },
                        readOnly=True,
                    ),
                ],
            ),
        ]
    )
)


@app.callback(
    Output(component_id="result", component_property="value"),
    [
        Input(component_id="input1", component_property="value"),
        Input(component_id="input2", component_property="value"),
        Input(component_id="check", component_property="n_clicks")
     ],
)
def search(input1,input2,n_clicks):
    if n_clicks == 0:
        global c
        c = n_clicks
        result = "see here"
    elif n_clicks != c:
        result = infer(s_model, tokenize, input1, input2, device)
        if result == 0:
            result = "Entailment"
        elif result == 1:
            result = "Neutral"
        elif result == 2:
            result = "Contradiction"
    else:
        result = "refresh"

    return result

# Run the app
if __name__ == "__main__":
    app.run_server()
