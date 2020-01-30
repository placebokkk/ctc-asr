import os
import argparse
import json

import torch
import torch.nn.functional as F

from dataset import KaldiFeatureReader
from model import Net


def load_prior(prior_file):
    a = None
    with open(prior_file) as f:
        s = f.read().strip()
        a = [int(i) for i in s.split()[1:-1]]
        a = torch.FloatTensor(a)
    return torch.log((a/a.sum()))


def extract(dataset, model, prior_file, out_file):
    log_priors = load_prior(prior_file)
    with open(out_file, 'w') as f:
        for _, (keys, xs, xlens) in enumerate(dataset):
            outputs = model(xs)
            outputs = F.log_softmax(outputs, dim=2)
            for i in range(len(keys)):
                k = keys[i]
                output = outputs[i]
                output = output - log_priors
                xlen = xlens[i]
                f.write('{} [\n'.format(k))
                for t in range(xlen-1):
                    formatted_list = ['%.5f' %
                                      elem for elem in output[t].tolist()]
                    f.write(' '.join(formatted_list))
                    f.write('\n')
                formatted_list = ['%.5f' % elem for elem in output[t].tolist()]
                f.write(' '.join(formatted_list))
                f.write(' ]\n'.format(k))


# On CPU
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')

    parser.add_argument('--model_conf', required=True, help='model config')
    parser.add_argument('--prior_file', required=True, help='prior probs')
    parser.add_argument('--data_dir', required=True, help='data dir')
    parser.add_argument('--model_file', help='model path')
    parser.add_argument(
        '--out_file', help='the prob will be output to this file')

    args = parser.parse_args()

    with open(args.model_conf) as fin:
        json_string = fin.read()
    model_paras = json.loads(json_string)

    feat_scp = os.path.join(args.data_dir, 'feats.sort.scp')
    utt2spk = os.path.join(args.data_dir, 'utt2spk')
    cmvn_scp = os.path.join(args.data_dir, 'cmvn.scp')
    dataset = KaldiFeatureReader(feat_scp, utt2spk, cmvn_scp, 1)
    # model_path='/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/model2/epoch_5.pt'
    checkpoint = torch.load(args.model_file)

    model = Net(model_paras)
    model.load_state_dict(checkpoint['model'])
    extract(dataset, model, args.prior_file, args.out_file)
