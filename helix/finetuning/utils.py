from Bio import SeqIO
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


def spearman(y_pred, y_true):
    import numpy as np
    from scipy.stats import spearmanr
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]


def compute_stat(sr):
    import numpy as np
    sr = np.asarray(sr)
    mean = np.mean(sr)
    std = np.std(sr)
    return mean, std


def compute_score(model, seq, mask, wt, pos, tokenizer):
    '''
    compute mutational proxy using masked marginal probability
    :param seq:mutant seq
    :param mask:attention mask for input seq
    :param wt: wild type sequence
    :param pos:mutant position
    :return:
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    import torch
    device = seq.device

    mask_seq = seq.clone()
    m_id = tokenizer.mask_token_id

    batch_size = int(seq.shape[0])
    for i in range(batch_size):
        mut_pos = pos[i]
        mask_seq[i, mut_pos+1] = m_id

    out = model(mask_seq, mask, output_hidden_states=True)
    logits = out.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(batch_size)
    scores = scores.to(device)

    for i in range(batch_size):

        mut_pos = pos[i]
        score_i = log_probs[i]
        wt_i = wt[i]
        seq_i = seq[i]
        scores[i] = torch.sum(score_i[mut_pos+1, seq_i[mut_pos+1]]) - \
            torch.sum(score_i[mut_pos+1, wt_i[mut_pos+1]])

    return scores, logits


def BT_loss(scores, golden_score):
    import torch
    loss = torch.tensor(0.)
    loss = loss.cuda()
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if golden_score[i] > golden_score[j]:
                loss += torch.log(1+torch.exp(scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-scores[j]))
    return loss


def KLloss(logits, logits_reg, seq, att_mask):
    import torch
    creterion_reg = torch.nn.KLDivLoss(reduction='mean')
    batch_size = int(seq.shape[0])

    loss = torch.tensor(0.)
    loss = loss.cuda()
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)
    for i in range(batch_size):

        probs_i = probs[i]
        probs_reg_i = probs_reg[i]

        seq_len = torch.sum(att_mask[i])

        reg = probs_reg_i[torch.arange(0, seq_len), seq[i, :seq_len]]
        pred = probs_i[torch.arange(0, seq_len), seq[i, :seq_len]]

        loss += creterion_reg(reg.log(), pred)
    return loss


class MutationDataset(Dataset):
    def __init__(self, data, fname, tokenizer, sep_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = sep_len
        self.seq, self.attention_mask = tokenizer(list(self.data['seq']), padding='max_length',
                                                  truncation=True,
                                                  max_length=self.seq_len).values()
        wt_path = os.path.join('/confit/data', fname, 'wt.fasta')
        for seq_record in SeqIO.parse(wt_path, "fasta"):
            wt = str(seq_record.seq)
        target = [wt]*len(self.data)
        self.target, self.tgt_mask = tokenizer(target, padding='max_length', truncation=True,
                                               max_length=self.seq_len).values()
        self.score = torch.tensor(np.array(self.data['log_fitness']))
        self.pid = np.asarray(data['PID'])

        if type(list(self.data['mutated_position'])[0]) != str:
            self.position = [[u] for u in self.data['mutated_position']]

        else:

            temp = [u.split(',') for u in self.data['mutated_position']]
            self.position = []
            for u in temp:
                pos = [int(v) for v in u]
                self.position.append(pos)

    def __getitem__(self, idx):
        return [self.seq[idx], self.attention_mask[idx], self.target[idx], self.tgt_mask[idx], self.position[idx], self.score[idx], self.pid[idx]]

    def __len__(self):
        return len(self.score)

    def collate_fn(self, data):
        seq = torch.tensor(np.array([u[0] for u in data]))
        att_mask = torch.tensor(np.array([u[1] for u in data]))
        tgt = torch.tensor(np.array([u[2] for u in data]))
        tgt_mask = torch.tensor(np.array([u[3] for u in data]))
        pos = [torch.tensor(u[4]) for u in data]
        score = torch.tensor(np.array([u[5]
                             for u in data]), dtype=torch.float32)
        pid = torch.tensor(np.array([u[6] for u in data]))
        return seq, att_mask, tgt, tgt_mask, pos, score, pid


def sample_data(dataset_name, seed, shot, frac=0.2):
    '''
    sample the train data and test data
    :param seed: sample seed
    :param frac: the fraction of testing data, default to 0.2
    :param shot: the size of training data
    '''

    data = pd.read_csv(f'/confit/data/{dataset_name}/data.csv', index_col=0)
    test_data = data.sample(frac=frac, random_state=seed)
    train_data = data.drop(test_data.index)
    kshot_data = train_data.sample(n=shot, random_state=seed)
    assert len(kshot_data) == shot, (
        f'expected {shot} train examples, received {len(train_data)}')

    kshot_data.to_csv(f'/confit/data/{dataset_name}/train.csv')
    test_data.to_csv(f'/confit/data/{dataset_name}/test.csv')


def split_train(dataset_name):
    '''
    five equal split training data, one of which will be used as validation set when training ConFit
    '''
    train = pd.read_csv(f'/confit/data/{dataset_name}/train.csv', index_col=0)
    tlen = int(np.ceil(len(train) / 5))
    start = 0
    for i in range(1, 5):
        csv = train[start:start + tlen]
        start += tlen
        csv.to_csv(f'/confit/data/{dataset_name}/train_{i}.csv')
    csv = train[start:]
    csv.to_csv(f'/confit/data/{dataset_name}/train_{5}.csv')


def evaluate(model, testloader, tokenizer, accelerator, istest=False):
    model.eval()
    seq_list = []
    score_list = []
    gscore_list = []
    with torch.no_grad():
        for step, data in enumerate(testloader):
            seq, mask = data[0], data[1]
            wt, wt_mask = data[2], data[3]
            pos = data[4]
            golden_score = data[5]
            pid = data[6]
            if istest:
                pid = pid.cuda()
                pid = accelerator.gather(pid)
                for s in pid:
                    seq_list.append(s.cpu())

            score, logits = compute_score(model, seq, mask, wt, pos, tokenizer)

            score = score.cuda()
            score = accelerator.gather(score)
            golden_score = accelerator.gather(golden_score)
            score = np.asarray(score.cpu())
            golden_score = np.asarray(golden_score.cpu())
            score_list.extend(score)
            gscore_list.extend(golden_score)
    score_list = np.asarray(score_list)
    gscore_list = np.asarray(gscore_list)
    sr = spearman(score_list, gscore_list)

    if istest:
        seq_list = np.asarray(seq_list)

        return sr, score_list, seq_list
    else:
        return sr
