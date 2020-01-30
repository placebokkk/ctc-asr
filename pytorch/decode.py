import argparse

import commentjson
import torch

from dataset import KaldiReader
from model import Net


def decode(dataset, model):
    for i, (keys, xs, xlens) in enumerate(dataset):
        outputs = model(xs)
        logp, pred = torch.max(outputs, dim=2)
        print(pred, logp.sum())


# On CPU
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')

    parser.add_argument('--model_conf', required=True, help='model config')
    parser.add_argument('--model_path', help='model path')

    args = parser.parse_args()

    with open(args.model_conf) as fin:
        json_string = fin.read()
    model_paras = commentjson.loads(json_string)

    feat_scp = os.path.join(args.train_data_dir, 'feats.scp')
    utt2spk = os.path.join(args.train_data_dir, 'utt2spk')
    cmvn_scp = os.path.join(args.train_data_dir, 'cmvn.scp')
    dataset = KaldiFeatureReader(feat_scp, utt2spk, cmvn_scp, 1)
    # model_path='/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/model2/epoch_5.pt'
    checkpoint = torch.load(args.model_path)

    model = Net(model_paras)
    model.load_state_dict(checkpoint['model'])
    torch.set_printoptions(profile="full")
    decode(dataset, model)
