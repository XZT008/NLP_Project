import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model
from torch.distributions import Categorical
from rouge import Rouge
from numpy import random
from util import config
from tqdm import tqdm

def train(model, dataloader, optimizer, epoch):
    model.train()
    loss = []
    for i, batch in tqdm(enumerate(dataloader)):
        doc_input, doc_extend_vocab, sum_input, sum_target, input_masks, doc_len, sum_len, vocab_pad = \
        batch.doc_input, batch.doc_extend_vocab, batch.sum_input, batch.sum_target, \
        batch.input_masks, batch.doc_len, batch.sum_len, batch.vocab_pad
        enc_batch = model.embeds(doc_input)  # Get embeddings for encoder input
        enc_out, enc_hidden = model.encoder(enc_batch, doc_len)
        ct_e = torch.zeros(doc_len.size(0), 2 * config.hidden_dim).cuda()

        max_dec_len = torch.max(sum_len)
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])  # Decoder hidden states
        x_t = (torch.LongTensor(len(enc_out)).fill_(2)).cuda()  # Input to the decoder
        prev_s = None  # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        for t in range(max_dec_len):
            use_gound_truth = (torch.rand(len(enc_out)) > 0.25).long().cuda()
            x_t = use_gound_truth * sum_input[:, t] + (1 - use_gound_truth) * x_t
            x_t = model.embeds(x_t)
            final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = model.decoder(x_t, s_t, enc_out, input_masks,
                                                                                  ct_e, vocab_pad,
                                                                                  doc_extend_vocab,
                                                                                  sum_temporal_srcs, prev_s)
            target = sum_target[:, t]
            log_probs = torch.log(final_dist + config.eps)
            step_loss = F.nll_loss(log_probs, target.long(), reduction="none", ignore_index=0)
            step_losses.append(step_loss)
            x_t = torch.multinomial(final_dist, 1).squeeze()  # Sample words from final distribution which can be used as input in next time step
            is_oov = (x_t >= 50004).long()  # Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * 1  # Replace OOVs with [UNK] token

        losses = torch.sum(torch.stack(step_losses, 1), 1)  # unnormalized losses for each example in the batch; (batch_size)
        batch_avg_loss = losses / sum_len.cuda()  # Normalized losses; (batch_size)
        mle_loss = torch.mean(batch_avg_loss)
        loss.append(mle_loss.item())
        optimizer.zero_grad()
        mle_loss.backward()
        optimizer.step()

        if i%50==0 and i != 0:
            print(f'At itr:{i}, average NLL is {sum(loss[i-50:i]) / 50}')

    print(f'At epoch:{epoch}, average NLL is {sum(loss)/len(loss)}')

def train_loop(epochs, train_loader, model, optimizer):
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)
        torch.save(model, f'saved/saved_model/{epoch}.pt')
