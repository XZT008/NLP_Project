import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model
from torch.distributions import Categorical
from rouge import Rouge
from numpy import random
from util import config
from tqdm import tqdm
import pickle

# completely teacher forcing
def loss_mle(model, enc_out, enc_hidden, batch, batch_size):
    max_len = torch.max(batch.sum_len)
    ct_e = torch.zeros(batch_size, 2 * config.hidden_dim).cuda()
    input_seq = batch.sum_input
    target_seq = batch.sum_target

    prev_s = None
    sum_temporal_srcs = None
    loss_l = []
    for t in range(max_len):
        input = input_seq[:, t]
        target = target_seq[:, t]
        embeded_input = model.embeds(input)
        pred, enc_hidden, ct_e, sum_temporal_srcs, prev_s = model.decoder(embeded_input, enc_hidden, enc_out,
                                                                          batch.input_masks, ct_e,
                                                                          batch.vocab_pad, batch.doc_extend_vocab,
                                                                          sum_temporal_srcs, prev_s)
        loss = F.nll_loss(torch.log(pred + config.eps), target, reduction='none', ignore_index=0)
        loss_l.append(loss)

    losses = torch.sum(torch.stack(loss_l, 1))
    mle_loss = losses / torch.sum(batch.sum_len).cuda()
    return mle_loss


def loss_rl_simple():
    pass


def loss_rl(model, enc_out, enc_hidden, batch, batch_size, id2words, use_simple=True):
    # get T
    max_len = torch.max(batch.sum_len)

    # get baseline R(y_*), where y_* is sampled by greedy approach, so it is deterministic
    scores, _, _ = get_greedy_sample(model, enc_out, enc_hidden, batch, batch_size, id2words)
    greedy_scores = torch.tensor([score["rouge-l"]["f"] for score in scores]).cuda()

    # get one sample from categorical dist
    samples, log_probs, log_probs_unnormed, mask, enc_hidden_l_0, enc_hidden_l_1, ct_e_l, sum_temporal_srcs_l, prev_s = get_sample(model, enc_out, enc_hidden, batch, batch_size)
    if use_simple:
        targets = batch.sum_target
        oovs = batch.oovs
        sample_scores, sample_sent, target_sent = get_scores(samples, targets, oovs, id2words)
        sample_scores = torch.tensor([score["rouge-l"]["f"] for score in sample_scores]).cuda()
        sample_rewards = torch.mean(sample_scores)
        rl_loss = torch.mean(-(sample_scores - greedy_scores) * log_probs)
        return rl_loss, sample_rewards
    else:
        targets = batch.sum_target
        oovs = batch.oovs
        sample_scores, sample_sent, target_sent = get_scores(samples, targets, oovs, id2words)
        sample_scores = torch.tensor([score["rouge-l"]["f"] for score in sample_scores]).cuda()
        sample_rewards = torch.mean(sample_scores)
        reward_matrix = rollout(model, enc_out, batch, batch_size, samples, enc_hidden_l_0, enc_hidden_l_1, ct_e_l, sum_temporal_srcs_l, prev_s, id2words)
        reward_matrix[:, -1] = sample_scores
        reward_matrix = reward_matrix - greedy_scores.unsqueeze(1)
        norm = torch.sum(mask, dim=1)
        log_probs = torch.sum(log_probs_unnormed * reward_matrix * mask, dim=1) / (norm + 1)
        rl_loss = torch.mean(-log_probs)
        return rl_loss, sample_rewards



# rollout from timestep t. 0<t<max_len, returns reward matrix
def rollout(model, enc_out, batch, batch_size, samples, enc_hidden_l_0, enc_hidden_l_1, ct_e_l, sum_temporal_srcs_l, prev_s_l,
            id2words):
    max_len = torch.max(batch.sum_len)
    reward_matrix = torch.zeros(batch_size, max_len.item()).cuda()
    with torch.no_grad():
        for t in range(max_len.item()-1):
            prev_sampled = samples[:, :t+1]
            prev_sampled = torch.where(prev_sampled>=50004, 1, prev_sampled)
            prev_s = prev_s_l[:, :t+1, :]
            enc_hidden = (enc_hidden_l_0[t], enc_hidden_l_1[t])
            ct_e = ct_e_l[t]
            sum_temporal_srcs = sum_temporal_srcs_l[t]
            steps_to_go = max_len.item()-t-1
            rollout_samples = rollout_sample(model, prev_sampled, prev_s, enc_hidden, ct_e, sum_temporal_srcs, steps_to_go,
                                             batch, enc_out)

            targets = batch.sum_target
            oovs = batch.oovs
            sample_scores, sample_sent, target_sent = get_scores(rollout_samples, targets, oovs, id2words)
            reward_matrix[:, t] = torch.tensor([score["rouge-l"]["f"] for score in sample_scores])

    return reward_matrix


def rollout_sample(model, prev_sampled, prev_s, enc_hidden, ct_e, sum_temporal_srcs, steps_to_go, batch, enc_out):
    input = prev_sampled[:, -1]
    samples = []

    with torch.no_grad():
        for t in range(steps_to_go):
            embeded_input = model.embeds(input)
            pred, enc_hidden, ct_e, sum_temporal_srcs, prev_s = model.decoder(embeded_input, enc_hidden, enc_out,
                                                                              batch.input_masks, ct_e,
                                                                              batch.vocab_pad, batch.doc_extend_vocab,
                                                                              sum_temporal_srcs, prev_s)
            cat_dist = torch.distributions.categorical.Categorical(probs=pred)
            input = cat_dist.sample()
            samples.append(input)
            input = torch.where(input >= 50004, 1, input)

    samples = torch.stack(samples, 1)
    samples = torch.cat((prev_sampled, samples), 1)
    return samples

# get sample from categorial with log prob and mask
def get_sample(model, enc_out, enc_hidden, batch, batch_size):
    max_len = torch.max(batch.sum_len)
    ct_e = torch.zeros(batch_size, 2 * config.hidden_dim).cuda()
    input_seq = batch.sum_input

    prev_s = None
    sum_temporal_srcs = None
    input = input_seq[:, 0]
    samples = []
    log_probs = []


    enc_hidden_l_0 = []
    enc_hidden_l_1 = []
    ct_e_l = []
    sum_temporal_srcs_l = []


    for t in range(max_len):
        embeded_input = model.embeds(input)
        pred, enc_hidden, ct_e, sum_temporal_srcs, prev_s = model.decoder(embeded_input, enc_hidden, enc_out,
                                                                          batch.input_masks, ct_e,
                                                                          batch.vocab_pad, batch.doc_extend_vocab,
                                                                          sum_temporal_srcs, prev_s)
        enc_hidden_l_0.append(enc_hidden[0])
        enc_hidden_l_1.append(enc_hidden[1])
        ct_e_l.append(ct_e)
        sum_temporal_srcs_l.append(sum_temporal_srcs)

        cat_dist = torch.distributions.categorical.Categorical(probs=pred)
        input = cat_dist.sample()
        log_prob = cat_dist.log_prob(input)
        log_probs.append(log_prob)
        samples.append(input)
        input = torch.where(input >= 50004, 1, input)

    enc_hidden_l_0 = torch.stack(enc_hidden_l_0, dim=0)
    enc_hidden_l_1 = torch.stack(enc_hidden_l_1, dim=0)
    ct_e_l = torch.stack(ct_e_l, dim=0)
    sum_temporal_srcs_l = torch.stack(sum_temporal_srcs_l, dim=0)


    samples = torch.stack(samples, 1)
    log_probs = torch.stack(log_probs, dim=1)
    log_probs_unnormed = log_probs
    mask = torch.ones_like(log_probs)
    for i in range(batch_size):
        indices = torch.nonzero((samples[i] == 3), as_tuple=True)[0]
        if indices.nelement() == 0:
            continue
        else:
            index, _ = torch.sort(indices)
            zero_indices = torch.arange(index[0], max_len).cuda()
            mask[i].index_fill_(0, zero_indices, 0)

    norm = torch.sum(mask, dim=1)
    log_probs = torch.sum(log_probs*mask, dim=1) / (norm+1)
    return samples, log_probs, log_probs_unnormed, mask, enc_hidden_l_0, enc_hidden_l_1, ct_e_l, sum_temporal_srcs_l, prev_s


def get_scores(greedy_samples, target, oovs, id2words):
    sample_sent = []
    target_sent = []
    for g, t, oov in zip(greedy_samples, target, oovs):
        g = g.tolist()
        t = t.tolist()
        try:
            g_index = g.index(3)
            g = g[:g_index]
        except ValueError:
            g = g

        try:
            t_index = t.index(3)
            t = t[:t_index]
        except ValueError:
            t = t

        tmp_vocab = id2words + oov
        g_sent = [tmp_vocab[i] for i in g]
        t_sent = [tmp_vocab[i] for i in t]

        if len(g_sent) < 2:
            g_sent = ['XXX']
        if len(t_sent) < 2:
            t_sent = ['XXX']
        g_sent = ' '.join(g_sent)
        t_sent = ' '.join(t_sent)
        sample_sent.append(g_sent)
        target_sent.append(t_sent)

    rouge = Rouge()
    scores = rouge.get_scores(sample_sent, target_sent)
    return scores, sample_sent, target_sent


def get_greedy_sample(model, enc_out, enc_hidden, batch, batch_size, id2words):
    max_len = torch.max(batch.sum_len)
    ct_e = torch.zeros(batch_size, 2 * config.hidden_dim).cuda()
    input_seq = batch.sum_input

    prev_s = None
    sum_temporal_srcs = None
    input = input_seq[:, 0]
    samples = []
    with torch.no_grad():
        for t in range(max_len):
            embeded_input = model.embeds(input)
            pred, enc_hidden, ct_e, sum_temporal_srcs, prev_s = model.decoder(embeded_input, enc_hidden, enc_out,
                                                                              batch.input_masks, ct_e,
                                                                              batch.vocab_pad, batch.doc_extend_vocab,
                                                                              sum_temporal_srcs, prev_s)
            input = torch.argmax(pred, 1)
            samples.append(input)
            input = torch.where(input >= 50004, 1, input)

    samples = torch.stack(samples, 1)
    targets = batch.sum_target
    oovs = batch.oovs
    scores, sample_sent, target_sent = get_scores(samples.detach().cpu().numpy(), targets.detach().cpu().numpy(),
                                                         oovs, id2words)
    return scores, sample_sent, target_sent


def train_loop(epochs, train_loader, model, optimizer, id2words):
    model.train()
    for epoch in range(epochs):
        mle_losses = []
        simple_rl_losses = []
        rl_losses = []

        rewards = []
        for i, batch in tqdm(enumerate(train_loader)):
            doc_input, doc_extend_vocab, sum_input, sum_target, input_masks, doc_len, sum_len, vocab_pad = \
                batch.doc_input, batch.doc_extend_vocab, batch.sum_input, batch.sum_target, \
                batch.input_masks, batch.doc_len, batch.sum_len, batch.vocab_pad
            batch_size = doc_len.size(0)
            enc_batch = model.embeds(doc_input)
            enc_out, enc_hidden = model.encoder(enc_batch, doc_len)
            mle_loss = loss_mle(model, enc_out, enc_hidden, batch, batch_size)
            simple_rl_loss, reward = loss_rl(model, enc_out, enc_hidden, batch, batch_size, id2words, use_simple=True)
            simple_rl_losses.append(simple_rl_loss.item())
            if i % 30 == 0:
                rl_loss, reward = loss_rl(model, enc_out, enc_hidden, batch, batch_size, id2words, use_simple=False)
                rl_losses.append(rl_loss.item())
                loss = 0.25*mle_loss + 0.25*simple_rl_loss + 0.5*rl_loss
            else:
                loss = 0.25*mle_loss + 0.75*simple_rl_loss
            mle_losses.append(mle_loss.item())
            rewards.append(reward.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%30 == 0 and i!= 0:
                print(f'At itr:{i}, avg mle loss:{sum(mle_losses[i-30:i]) / 30}, avg simple rl loss:{sum(simple_rl_losses[i-30:i]) / 30}, avg rl loss:{sum(rl_losses) / len(rl_losses)}, avg reward: {sum(rewards[i-30:i]) / 30}')

            if i % 1000 == 0 and i!=0:
                with open(f'mle_{epoch}.pkl', 'wb') as f:
                    pickle.dump(mle_losses, f)
                with open(f'sim_rl_{epoch}.pkl', 'wb') as f:
                    pickle.dump(simple_rl_losses, f)
                with open(f'rewards_{epoch}.pkl', 'wb') as f:
                    pickle.dump(rewards, f)
                with open(f'rl_{epoch}.pkl', 'wb') as f:
                    pickle.dump(rl_losses, f)

        print(f'At epoch:{epoch}, avg mle loss:{sum(mle_losses)/len(mle_losses)}, avg simple rl loss:{sum(simple_rl_losses)/len(simple_rl_losses)}, avg rl loss:{sum(rl_losses)/len(rl_losses)}, avg reward: {sum(rewards)/len(rewards)}')
        #torch.save(model, f'saved/saved_model/{epoch}.pt')



def train_loop_rl_only(epochs, train_loader, model, optimizer, id2words):
    model.train()
    for epoch in range(epochs):
        simple_rl_losses = []
        rl_losses = []
        rewards = []

        for i, batch in tqdm(enumerate(train_loader)):
            doc_input, doc_extend_vocab, sum_input, sum_target, input_masks, doc_len, sum_len, vocab_pad = \
                batch.doc_input, batch.doc_extend_vocab, batch.sum_input, batch.sum_target, \
                batch.input_masks, batch.doc_len, batch.sum_len, batch.vocab_pad
            batch_size = doc_len.size(0)
            enc_batch = model.embeds(doc_input)
            enc_out, enc_hidden = model.encoder(enc_batch, doc_len)
            #mle_loss = loss_mle(model, enc_out, enc_hidden, batch, batch_size)
            simple_rl_loss, reward = loss_rl(model, enc_out, enc_hidden, batch, batch_size, id2words, use_simple=True)
            simple_rl_losses.append(simple_rl_loss.item())
            if i % 30 == 0:
                rl_loss, reward = loss_rl(model, enc_out, enc_hidden, batch, batch_size, id2words, use_simple=False)
                rl_losses.append(rl_loss.item())
                loss = 0.5*simple_rl_loss + 0.5*rl_loss
            else:
                loss = simple_rl_loss
            rewards.append(reward.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%30 == 0 and i!= 0:
                print(f'At itr:{i}, avg simple rl loss:{sum(simple_rl_losses[i-30:i]) / 30}, avg rl loss:{sum(rl_losses) / len(rl_losses)}, avg reward: {sum(rewards[i-30:i]) / 30}')

        print(f'At epoch:{epoch}, avg simple rl loss:{sum(simple_rl_losses)/len(simple_rl_losses)}, avg rl loss:{sum(rl_losses)/len(rl_losses)}, avg reward: {sum(rewards)/len(rewards)}')
        with open(f'sim_rl_{epoch}.pkl', 'wb') as f:
            pickle.dump(simple_rl_losses, f)
        with open(f'rewards_{epoch}.pkl', 'wb') as f:
            pickle.dump(rewards, f)
        with open(f'rl_{epoch}.pkl', 'wb') as f:
            pickle.dump(rl_losses, f)
        torch.save(model, f'saved/saved_model/rl_{epoch}.pt')
