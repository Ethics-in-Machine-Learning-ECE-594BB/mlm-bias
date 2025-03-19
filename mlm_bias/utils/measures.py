#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import numpy as np

# def get_mlm_output(model, inputs):
#     with torch.no_grad():
#         output = model(inputs, return_dict=True)
#     return output
def get_mlm_output(model, inputs):
    device = next(model.parameters()).device  # Get the model's device

    if isinstance(inputs, torch.Tensor):  # Ensure token_ids are on the correct device
        inputs = inputs.to(device)

    output = model(inputs, return_dict=True)
    return output
import torch
import numpy as np

@torch.no_grad()
def compute_crr_dp(
    model,
    token_ids,
    mask_token_index,
    masked_tok,
    measures=['crr','dp'],
    attention=True,
    log_softmax=False
):
    device = next(model.parameters()).device  # Ensure tensors are on the correct device
    token_ids = token_ids.to(device)
    
    if log_softmax:
        f_softmax = torch.nn.LogSoftmax(dim=1)
    else:
        f_softmax = torch.nn.Softmax(dim=1)

    output = get_mlm_output(model, token_ids)
    token_logits = output.logits.to(device)
    
    attn = output.attentions
    attentions = torch.mean(torch.cat(attn, 0), 0).to(device)
    attentions_avg = torch.mean(attentions, 0).to(device)
    attentions_avg_tok = torch.mean(attentions_avg, 0)[1:-1].to(device)

    mask_token_logits = token_logits.squeeze(0)[mask_token_index].to(device)
    mask_token_probs = f_softmax(mask_token_logits).to(device)
    
    top_toks = torch.topk(mask_token_probs, mask_token_probs.shape[1], dim=1).indices[0].tolist()
    top_token = top_toks[0]
    top_token_score = mask_token_probs[:, top_token].item()

    tok_inds = list(range(mask_token_probs.shape[1]))
    masked_token_index = tok_inds.index(masked_tok)
    masked_token_score = mask_token_probs[:, masked_token_index].item()
    masked_token_rank = top_toks.index(masked_tok) + 1

    token_j = {
        "token_id": masked_tok,
        "score": masked_token_score,
        "rank": masked_token_rank
    }

    if 'crr' in measures:
        crr = (1 - (1 / masked_token_rank))
    if 'dp' in measures:
        dp = (np.log(top_token_score) - np.log(masked_token_score))

    if attention:
        attw = np.mean([att.tolist() for att in [attentions_avg_tok]], axis=1)
        if 'crr' in measures:
            crr_attns = (attw[0] * (1 - np.log(1 / masked_token_rank)))
        if 'dp' in measures:
            dp_attns = (attw[0] * (np.log(top_token_score) - np.log(masked_token_score)))

    if 'crr' in measures:
        token_j['crr'] = crr
        token_j['crra'] = crr_attns
    if 'dp' in measures:
        token_j['dp'] = dp
        token_j['dpa'] = dp_attns

    return {
        "prediction": {
            "token_id": top_token,
            "score": top_token_score,
            "rank": 1
        },
        "masked_token": token_j
    }

@torch.no_grad()
def compute_aul(model, token_ids, attention=True, log_softmax=True):
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)

    if log_softmax:
        f_softmax = torch.nn.LogSoftmax(dim=1)
    else:
        f_softmax = torch.nn.Softmax(dim=1)

    output = get_mlm_output(model, token_ids)
    logits = output.logits.squeeze(0).to(device)
    probs = f_softmax(logits).to(device)

    token_ids = token_ids.view(-1, 1).detach().to(device)
    token_probs = probs.gather(1, token_ids).to(device)[1:-1]

    if attention:
        attentions = torch.mean(torch.cat(output.attentions, 0), 0).to(device)
        averaged_attentions = torch.mean(attentions, 0).to(device)
        averaged_token_attentions = torch.mean(averaged_attentions, 0).to(device)
        token_probs_attns = token_probs.squeeze(1) * averaged_token_attentions[1:-1]

    sentence_log_prob = torch.mean(token_probs).item()
    sentence_log_prob_attns = torch.mean(token_probs_attns).item()

    sorted_indexes = torch.sort(probs, dim=1, descending=True)[1].to(device)
    ranks = torch.where(sorted_indexes == token_ids)[1] + 1
    ranks = ranks.tolist()

    return {
        "aul": sentence_log_prob,
        "aula": sentence_log_prob_attns,
        "ranks": ranks,
    }

@torch.no_grad()
def compute_csps(model, token_ids, spans, mask_id, log_softmax=True):
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)

    if log_softmax:
        f_softmax = torch.nn.LogSoftmax(dim=1)
    else:
        f_softmax = torch.nn.Softmax(dim=1)

    spans = spans[1:-1]
    masked_token_ids = token_ids.repeat(len(spans), 1).to(device)
    masked_token_ids[range(masked_token_ids.size(0)), spans] = mask_id

    hidden_states = get_mlm_output(model, masked_token_ids)[0].to(device)
    token_ids = token_ids.view(-1)[spans].to(device)

    probs = f_softmax(hidden_states[range(hidden_states.size(0)), spans, :]).to(device)
    span_probs = probs[range(hidden_states.size(0)), token_ids].to(device)

    score = torch.sum(span_probs).item()

    sorted_indexes = torch.sort(probs, dim=1, descending=True)[1].to(device)
    ranks = torch.where(sorted_indexes == token_ids.view(-1, 1))[1] + 1
    ranks = ranks.tolist()
    # if len(ranks)%1 == 0:
    #     print(ranks)    
    return {
        "csps": score,
        "ranks": ranks,
    }

# @torch.no_grad()
# def compute_csps(model, tokenizer, token_ids, spans, mask_id, log_softmax=True):

#     device = next(model.parameters()).device
#     token_ids = token_ids.to(device)

#     if log_softmax:
#         f_softmax = torch.nn.LogSoftmax(dim=1)
#     else:
#         f_softmax = torch.nn.Softmax(dim=1)

#     spans = spans[1:-1]  # Remove special tokens like [CLS] and [SEP]
    
#     # # Calculate percentage of masked tokens
#     # total_tokens = token_ids.shape[1]
#     # num_masked = len(spans)
#     # masked_percentage = (num_masked / total_tokens) * 100

#     # print(f"\n--- Masking Debug Info ---")
#     # print(f"Total Tokens: {total_tokens}, Masked Tokens: {num_masked} ({masked_percentage:.2f}%)")

#     masked_token_ids = token_ids.repeat(len(spans), 1).to(device)
#     masked_token_ids[range(masked_token_ids.size(0)), spans] = mask_id  # Replace words with [MASK]

#     # Decode masked sentence
#     # masked_sentence = model.config.tokenizer.decode(masked_token_ids[0].tolist())
#     # masked_sentence = self.tokenizer.decode(masked_token_ids[0].tolist())
#     masked_sentence = tokenizer.decode(masked_token_ids[0].tolist())

#     print(f"Masked Sentence: {masked_sentence}")
    
#     hidden_states = get_mlm_output(model, masked_token_ids)[0].to(device)  # Get MLM predictions
#     token_ids = token_ids.view(-1)[spans].to(device)  # Get original tokens that were masked

#     probs = f_softmax(hidden_states[range(hidden_states.size(0)), spans, :]).to(device)
#     span_probs = probs[range(hidden_states.size(0)), token_ids].to(device)  # Probabilities of correct words

#     # Compute total CSPS score
#     score = torch.sum(span_probs).item()

#     # Get ranking of the correct word among all vocabulary predictions
#     sorted_indexes = torch.sort(probs, dim=1, descending=True)[1].to(device)
#     ranks = torch.where(sorted_indexes == token_ids.view(-1, 1))[1] + 1
#     ranks = ranks.tolist()

#     # Print detailed debug info
#     print("\n--- CSPS Debug Info ---")
#     print("Masked Token Positions:", spans)

#     top_k = 5
#     for i, (prob, rank, span_idx) in enumerate(zip(span_probs.tolist(), ranks, spans)):
#         top_preds = [tokenizer.decode([idx.item()]) for idx in sorted_indexes[i, :top_k]]
# #         print(f"Position {span_idx}: Correct Word Probability = {prob:.6f}, Rank = {rank}")
#         print(f"Position {span_idx}: Correct Word Probability = {prob:.6f}, Rank = {rank}, Top Predictions: {top_preds}")
    
#     return {
#         "csps": score,  # CSPS score (sum of correct word probabilities)
#         "ranks": ranks,  # Rank of correct word among all predictions
#     }


# @torch.no_grad()
# def compute_csps(model, token_ids, spans, mask_id, log_softmax=True):
#     device = next(model.parameters()).device
#     token_ids = token_ids.to(device)

#     if log_softmax:
#         f_softmax = torch.nn.LogSoftmax(dim=1)
#     else:
#         f_softmax = torch.nn.Softmax(dim=1)

#     spans = spans[1:-1]  # Remove special tokens like [CLS] and [SEP]
#         # Calculate percentage of masked tokens
#     total_tokens = token_ids.shape[1]
#     num_masked = len(spans)
#     masked_percentage = (num_masked / total_tokens) * 100

#     print(f"\n--- Masking Debug Info ---")
#     print(f"Total Tokens: {total_tokens}, Masked Tokens: {num_masked} ({masked_percentage:.2f}%)")


#     masked_token_ids = token_ids.repeat(len(spans), 1).to(device)
#     masked_token_ids[range(masked_token_ids.size(0)), spans] = mask_id  # Replace words with [MASK]

#     hidden_states = get_mlm_output(model, masked_token_ids)[0].to(device)  # Get MLM predictions
#     token_ids = token_ids.view(-1)[spans].to(device)  # Get original tokens that were masked

#     probs = f_softmax(hidden_states[range(hidden_states.size(0)), spans, :]).to(device)
#     span_probs = probs[range(hidden_states.size(0)), token_ids].to(device)  # Probabilities of correct words

#     # Compute total CSPS score
#     score = torch.sum(span_probs).item()

#     # Get ranking of the correct word among all vocabulary predictions
#     sorted_indexes = torch.sort(probs, dim=1, descending=True)[1].to(device)
#     ranks = torch.where(sorted_indexes == token_ids.view(-1, 1))[1] + 1
#     ranks = ranks.tolist()

#     # Print debug info
#     print("\n--- CSPS Debug Info ---")
#     print("Masked Token Positions:", spans)
#     for i, (prob, rank, span_idx) in enumerate(zip(span_probs.tolist(), ranks, spans)):
#         print(f"Position {span_idx}: Correct Word Probability = {prob:.6f}, Rank = {rank}")

#     return {
#         "csps": score,  # CSPS score (sum of correct word probabilities)
#         "ranks": ranks,  # Rank of correct word among all predictions
#     }

@torch.no_grad()
def compute_sss(model, token_ids, spans, mask_id, log_softmax=True):
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)

    if log_softmax:
        f_softmax = torch.nn.LogSoftmax(dim=1)
    else:
        f_softmax = torch.nn.Softmax(dim=1)

    masked_token_ids = token_ids.clone().to(device)
    masked_token_ids[:, spans] = mask_id

    hidden_states = get_mlm_output(model, masked_token_ids)[0].squeeze(0).to(device)
    token_ids = token_ids.view(-1)[spans].to(device)

    probs = f_softmax(hidden_states)[spans].to(device)
    span_probs = probs[:, token_ids].to(device)

    score = torch.mean(span_probs).item()

    if probs.size(0) != 0:
        sorted_indexes = torch.sort(probs, dim=1, descending=True)[1].to(device)
        ranks = torch.where(sorted_indexes == token_ids.view(-1, 1))[1] + 1
        ranks = ranks.tolist()
    else:
        ranks = [-1]
    # print("SSS", ranks)
    return {
        "sss": score,
        "ranks": ranks,
    }
