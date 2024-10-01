import json
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
#from sklearn.metrics import silhouette_score
#from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification


#punct_tokenizer = AutoTokenizer.from_pretrained("SJ-Ray/Re-Punctuate")
punct_tokenizer = AutoTokenizer.from_pretrained("felflare/bert-restore-punctuation")
punct_model = AutoModelForTokenClassification.from_pretrained("felflare/bert-restore-punctuation").cuda()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()

def hms2secs(x):
    h, m, s = x.split(':')
    s = s.replace(',', '.')
    return 3600*float(h) + 60*float(m) + float(s)

epname = 'oltl-10-18-10'
with open(f'SummScreen/closed_captions/{epname}.json') as f:
    closed_captions = json.load(f)['captions']

prev_start = 0
tokens = []
token_times = []
cur_cap = closed_captions[0]
N = 15000
text = ''
for next_cap in closed_captions[1:]:
    if cur_cap[1].strip() != '':
        start, _ = cur_cap[0].split(' --> ')
        start = hms2secs(start)
        _, stop = cur_cap[0].split(' --> ')
        stop = hms2secs(stop)
        new_text = cur_cap[1].replace('mrs', 'm.r.s.').replace('mr','m.r.')
        text += new_text
        new_toks = punct_tokenizer(new_text, add_special_tokens=False).input_ids
        new_tok_times = np.linspace(start, stop, len(new_toks))
        tokens += new_toks
        token_times += new_tok_times.tolist()
    cur_cap = next_cap
cc_df = pd.DataFrame({'token':tokens, 'time':token_times})

puncts = ['up', 'none', '?up', '?', ';', ':up', ':', '.up', '.', '-', ',up', ',', '\',', '!up', '!']
tokens = tokens[:N]
y = punct_model(torch.tensor(tokens[:256])[None].cuda())
punct_preds = y.logits.argmax(axis=2).squeeze(0)
restored_text_toks = []
for insert_idx, pp_idx in list(enumerate(punct_preds)):
    pp = puncts[int(pp_idx.item())]
    if pp == 'none':
        restored_text_toks.append(punct_tokenizer.decode(tokens[insert_idx]))
        continue
    make_cap = pp.endswith('up')
    insertee_tok = pp.removesuffix('up')
    if make_cap:
        following_tokid = tokens[insert_idx]
        following_tok = punct_tokenizer.decode(following_tokid)
        following_tok_upper = following_tok[0].upper() + following_tok[1:]
        restored_text_toks.append(following_tok_upper)
        restored_text_toks.append(insertee_tok)
        upper_tokids = punct_tokenizer(following_tok_upper, add_special_tokens=False).input_ids
    elif insertee_tok != '':
        restored_text_toks.append(punct_tokenizer.decode(tokens[insert_idx]))
        restored_text_toks.append(insertee_tok)
    if insert_idx == 0:
        tok_time_insertee = token_times[0]
    elif insert_idx == len(token_times) - 1:
        tok_time_insertee = token_times[-1]
    else:
        tok_time_insertee = (token_times[insert_idx] + token_times[insert_idx+1])/2
        tok_time_insertee = token_times[0]
    token_times.insert(insert_idx, tok_time_insertee)

print(punct_tokenizer.decode(tokens))
print(' '.join(restored_text_toks))
breakpoint()

while True:
    padded_len = math.ceil(len(tokens)/25)*25
    padded_tokens = tokens + [tokenizer.eos_token_id]*(padded_len - len(tokens))
    assert len(padded_tokens) % 25 == 0
    input_ids = torch.tensor(padded_tokens).cuda().reshape(-1, 25)
    output = model(input_ids=input_ids, output_attentions=True)
    logits = output.logits.reshape(len(padded_tokens), -1)[:len(tokens)]
    mxs, argmxs = logits.max(axis=1)
    fs_id = tokenizer('.').input_ids[0]
    qmark_id = tokenizer('?').input_ids[0]
    punct_ids = (fs_id, qmark_id)
    mxsargmxs = [(mx,argmx,idx+1) for idx, (mx,argmx) in enumerate(zip(mxs.tolist(), argmxs.tolist())) if argmx in punct_ids and tokens[idx+1] not in punct_ids]
    if len(mxsargmxs)==0:
        break
    _, insert_tokid, insert_idx = max(mxsargmxs, key=lambda x:x[0])
    tokens.insert(insert_idx, insert_tokid)
    print(tokenizer.decode(tokens))
logit_topks = logits.topk(axis=1, k=1).indices
idxs_to_insert_fullstop = torch.argwhere((logit_topks==fullstop_tok_id).any(axis=1)).squeeze(axis=1).tolist()
idxs_to_insert_qmark = torch.argwhere((logit_topks==qmark_tok_id).any(axis=1)).squeeze(axis=1)
#idxs_to_insert_qmark = [x for x in idxs_to_insert_qmark.tolist() if x not in idxs_to_insert_fullstop]
idxs_to_insert_comma = torch.argwhere((logit_topks==comma_tok_id).any(axis=1)).squeeze(axis=1)
#idxs_to_insert_comma = [x for x in idxs_to_insert_comma.tolist() if x not in idxs_to_insert_fullstop+idxs_to_insert_qmark]
breakpoint()
a = list(zip(idxs_to_insert_fullstop, [fullstop_tok_id]*len(idxs_to_insert_fullstop)))
b = list(zip(idxs_to_insert_qmark, [qmark_tok_id]*len(idxs_to_insert_qmark)))
c = list(zip(idxs_to_insert_comma, [comma_tok_id]*len(idxs_to_insert_comma)))

for insert_idx, insertee in sorted(a+b+c, key=lambda x:x[0], reverse=True):
    tokens.insert(insert_idx+1, insertee)
    if insert_idx == 0:
        tok_time_insertee = token_times[0]
    elif insert_idx == len(token_times) - 1:
        tok_time_insertee = token_times[-1]
    else:
        tok_time_insertee = (token_times[insert_idx] + token_times[insert_idx+1])/2
        tok_time_insertee = token_times[0]
    token_times.insert(insert_idx+1, tok_time_insertee)

segment_ids = [0] + [i for i,x in enumerate(tokens) if x in (fullstop_tok_id, qmark_tok_id)] + [len(tokens)]
segments = [tokens[segment_ids[i]+1:segment_ids[i+1]+1] for i in range(1, len(segment_ids)-1)]

for seg in segments:
    print(tokenizer.decode(seg))

breakpoint()


