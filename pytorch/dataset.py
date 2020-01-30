import os

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import kaldi_io

# Select kaldi,
if 'KALDI_ROOT' in os.environ:
    print(os.environ['KALDI_ROOT'])
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    print("no KALDI_ROOT, set KALDI_ROOT to /export/maryland/binbinzhang/kaldi")
    os.environ['KALDI_ROOT'] = '/export/maryland/binbinzhang/kaldi'


class KaldiFeatureLabelReader():
    def __init__(self, feat_scp, label_file, utt2spk, cmvn_scp, batch_size=1):
        self.feat_scp = feat_scp
        self.label_file = label_file
        #self.feats_rspecifier = self.feat_scp
        self.feats_rspecifier = 'ark:copy-feats scp:{} ark:- | apply-cmvn --norm-vars=true --utt2spk=ark:{} scp:{} ark:- ark:- |\
 add-deltas --delta-order=2 ark:- ark:-|'.format(feat_scp, utt2spk, cmvn_scp)
        print(self.feats_rspecifier)
        self.labels = {}
        self.batch_size = batch_size
        with open(self.label_file) as f:
            for l in f:
                items = l.split()
                self.labels[items[0]] = np.array(list(map(int, items[1:])))

    def __len__(self):
        return len(self.labels)

    # CTCloss para format. xs and ys need to be padded as the longest lengtm
    def _format(self, keys, xs, ys):
        xs_lengths = torch.from_numpy(
            np.array([x.shape[0] for x in xs], dtype=np.int32))
        ys_lengths = torch.from_numpy(
            np.array([y.shape[0] for y in ys], dtype=np.int32))
        xs = pad_sequence([torch.from_numpy(x).float() for x in xs], True, 0.0)
        ys = pad_sequence([torch.from_numpy(y).int() for y in ys], True, -1)
        return keys, xs, ys, xs_lengths, ys_lengths

    def __iter__(self):
        batch_feats = []
        batch_labels = []
        batch_keys = []
        # for k, v in kaldi_io.read_mat_scp(self.feats_rspecifier):
        for k, v in kaldi_io.read_mat_ark(self.feats_rspecifier):
            if len(batch_feats) >= self.batch_size:
                yield self._format(batch_keys, batch_feats, batch_labels)
                batch_feats = []
                batch_labels = []
                batch_keys = []
            batch_feats.append(v)
            batch_labels.append(self.labels[k])
            batch_keys.append(k)
        yield self._format(batch_keys, batch_feats, batch_labels)


class KaldiFeatureReader():
    def __init__(self, feat_scp, utt2spk, cmvn_scp, batch_size=1):
        self.feat_scp = feat_scp
        #self.feats_rspecifier = self.feat_scp
        self.feats_rspecifier = 'ark:copy-feats scp:{} ark:- | apply-cmvn --norm-vars=true --utt2spk=ark:{} scp:{} ark:- ark:- |\
 add-deltas --delta-order=2 ark:- ark:-|'.format(feat_scp, utt2spk, cmvn_scp)
        print(self.feats_rspecifier)
        self.batch_size = batch_size
        self.number = 0
        with open(self.feat_scp) as f:
            for l in f:
                self.number += 1

    def __len__(self):
        return self.number

    # CTCloss para format. xs and ys need to be padded as the longest lengtm
    def _format(self, keys, xs):
        xs_lengths = torch.from_numpy(
            np.array([x.shape[0] for x in xs], dtype=np.int32))
        xs = pad_sequence([torch.from_numpy(x).float() for x in xs], True, 0.0)
        return keys, xs, xs_lengths

    def __iter__(self):
        batch_feats = []
        batch_keys = []
        # for k, v in kaldi_io.read_mat_scp(self.feats_rspecifier):
        for k, v in kaldi_io.read_mat_ark(self.feats_rspecifier):
            if len(batch_feats) >= self.batch_size:
                yield self._format(batch_keys, batch_feats)
                batch_feats = []
                batch_keys = []
            batch_feats.append(v)
            batch_keys.append(k)
        yield self._format(batch_keys, batch_feats)


if __name__ == "__main__":
    feat_scp = "/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/data/raw_fbank_cv.sample.scp"
    label_file = "/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/data/labels.sample.cv"
    cmvn_scp = "/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/data/cmvn.smaple.cv"
    utt2spk = "/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/data/utt2spk.cv"
    data_reader = KaldiFeatureLabelReader(
        feat_scp, label_file, utt2spk, cmvn_scp, 4)
    for i, (keys, xs, ys, xlen, ylen) in enumerate(data_reader):
        print(keys, xs.size(), ys.size(), xlen, ylen)
        # for b in range(len(xs)):
        #    for t in range(len(xs[b])):
        #        print(xs[b][t])
        print(ys[0])

    feat_scp = "/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/data/raw_fbank_cv.sample.scp"
    label_file = "/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/data/labels.sample.cv"
    cmvn_scp = "/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/data/cmvn.smaple.cv"
    utt2spk = "/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/data/utt2spk.cv"
    data_reader = KaldiFeatureReader(
        feat_scp, utt2spk, cmvn_scp, 4)
    for i, (keys, xs, xlen) in enumerate(data_reader):
        print(keys, xs.size(), xlen)
        # for b in range(len(xs)):
        #    for t in range(len(xs[b])):
        #        print(xs[b][t])
        print(ys[0])
