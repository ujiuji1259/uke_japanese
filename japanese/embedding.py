import os
import re
import string
import json
import pickle

import torch
import numpy as np

from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel


def encode_sentence(tokenizer, model, tokens):
    is_split = []
    input_tokens = ['[CLS]']
    for token in tokens:
        tmp = tokenizer.tokenize(token)
        
        if len(input_tokens) + len(tmp) >= 511:
            break
        else:
            input_tokens.extend(tmp)
            is_split.append(len(tmp))
    input_tokens += ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    
    input_ids = torch.LongTensor([input_ids])
    outputs = model(input_ids, output_hidden_states=True).last_hidden_state.detach().numpy()
    bertcls  = outputs[0, 0, :]
    o1 = outputs[0, :, :]
    cls_token = o1[0]
    
    tokens_emb = []
    i = 1
    for j in is_split:
        if j == 1:
            tokens_emb.append(o1[i])
            i += 1
        else:
            tokens_emb.append(sum(o1[i:i+j]) / j)
            # tokens_emb.append(np.max(np.array(o1[i: i+j]), axis=0))
            i += j
        # if i >= len(is_split):
        #     break
    assert len(tokens_emb) == len(is_split)
    return tokens_emb, bertcls, cls_token

def flat_list(l):
    return [x for ll in l for x in ll]

def encode_sentences(token_list, tokenizer, model):
    tokenizer.do_word_tokenize = False

    document_embeddings = []
    cnt = 0
    for tokens in token_list:
        tokens_emb, bertcls, cls_token = encode_sentence(tokenizer, model, tokens)

        document_embeddings.append({
            'document_id': cnt,
            'doc_cls': cls_token,
            'doc_bertcls': bertcls,
            "tokens": tokens_emb
        })
        cnt += 1

    return document_embeddings


def get_cadidate_embeddings(token_list, document_embeddings, tokens):
    document_feats = []
    cnt = 0
    for candidate_phrase, document_emb, each_tokens in zip(token_list, document_embeddings, tokens):
        sentence_emb = document_emb['tokens']
        
        tmp_embeddings = []
        tmp_candidate_phrase = []
        
        for tmp, (i, j) in candidate_phrase:
            if j<=i:
                continue
            if j >= len(sentence_emb):
                break
            # tmp_embeddings.append(sum(sentence_emb[i:j]) / (j-i))
            tmp_embeddings.append(np.max(np.array(sentence_emb[i:j]), axis=0))
            tmp_candidate_phrase.append(tmp)

        candidate_phrases_embeddings = tmp_embeddings
        candidate_phrases = tmp_candidate_phrase

        document_feats.append({
            'document_id': cnt,
            'tokens': each_tokens,
            'candidate_phrases': candidate_phrases,
            'candidate_phrases_embeddings': candidate_phrases_embeddings,
            # 'sentence_embeddings': document_emb['doc_bertcls'],
            'sentence_embeddings': document_emb['doc_cls'],
        })
        cnt += 1
    return document_feats
