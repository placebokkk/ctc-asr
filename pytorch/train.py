import os
import argparse
import json

import torch
from torch import nn, autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_value_

from dataset import KaldiFeatureLabelReader
from model import Net

# [TODO] use toke accuracy as cv metric


def cv(evalset, model):
    print("eval:")
    total_loss = 0.0
    device_type = 'cuda'
    device = torch.device(device_type)
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
    model.eval()
    num_total_utts = 0
    with torch.no_grad():
        for batch_id, (k, xs, ys, xlen, ylen) in enumerate(evalset):
            xs = xs.to(device)
            num_utts = ylen.size(0)
            num_total_utts += num_utts
            outputs = model(xs)
            outputs = F.log_softmax(outputs, dim=2)
            loss = ctc_loss(outputs.transpose(0, 1), ys, xlen, ylen)
            total_loss += loss
    return total_loss/num_total_utts


def train(dataset, evalset, epoch_num, model_paras, model_dir, last_model_path=None, use_cuda=False):
    # Choose Device
    device_type = 'cpu'
    if use_cuda:
        if torch.cuda.is_available():
            print('cuda available')
            device_type = 'cuda'
        else:
            print('no cuda available')
    else:
        print('not use cuda')
    device = torch.device(device_type)

    # Load Model
    model = Net(model_paras)
    if last_model_path:
        checkpoint = torch.load(last_model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        for param in model.parameters():
            torch.nn.init.uniform(param, -0.1, 0.1)
        save_model_path = os.path.join(model_dir, 'init.pt')
        print('Checkpoint: save init to {}'.format(save_model_path))
        state_dict = model.state_dict()
        torch.save(
            {
                'model': state_dict,
                'epoch': 0,
            }, save_model_path)

    model = model.to(device)

    # Set Optimizer Type
    optim_method = 'adam'
    if 'adam' == optim_method:
        print('Use Adam')
        learning_rate = 1e-4
        l2_regularize = 1e-5
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=l2_regularize)
    # sgd not work! It is hard to train the The LSTM weight
    if 'sgd' == optim_method:
        learning_rate = 4e-4
        momentum = 0.9
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum)

    # CTC
    use_pytorch_ctc = True
    if use_pytorch_ctc:
        ctc_loss = nn.CTCLoss(blank=0)
    else:
        import warpctc_pytorch as warp_ctc
        ctc_loss = warp_ctc.CTCLoss()

    # Start training
    print('Training')
    last_epoch_loss = 10000
    last_cv_loss = 10000

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        print('epoch {}'.format(epoch))
        epoch_loss = 0.0
        num_epoch_utts = 0
        model.train()
        for batch_id, (k, xs, ys, xlen, ylen) in enumerate(dataset):
            # Only xs need to device
            if use_pytorch_ctc:
                xs = xs.to(device)
            else:
                xs = xs.to(device)

            num_utts = ylen.size(0)
            num_epoch_utts += num_utts
            # forward
            outputs = model(xs)
            # ctc_loss need Batch size at axis 1, here use transpose(0, 1) to N,T,D -> T,N,D
            if use_pytorch_ctc:
                # Also support below ys format, which is same with warp-ctc
                # ignore_id=-1
                # ys = [y[y != ignore_id] for y in ys]  # parse padded ys
                # ys = torch.cat(ys).cpu().int()  # batch x olen
                outputs = F.log_softmax(outputs, dim=2)
                loss = ctc_loss(outputs.transpose(0, 1), ys, xlen, ylen)
                # buildin CTC use mean as default, so no need to divide by num_utts,
                #loss = loss / num_utts
            else:
                ignore_id = -1
                ys = [y[y != ignore_id] for y in ys]  # parse padded ys
                ys = torch.cat(ys)  # batch x olen
                outputs = outputs.transpose(0, 1).contiguous()
                outputs.requires_grad_(True)
                loss = ctc_loss(outputs, ys, xlen, ylen)
                loss = torch.mean(loss)

            # Reset the gradients
            optimizer.zero_grad()
            # BackWard
            loss.backward()
            # Clip gradients to avoid too large value
            clip = 5
            clip_grad_value_(model.parameters(), clip)
            # norm=200
            #nn.utils.clip_grad_norm_(model.parameters(), norm)
            # Do weight update
            optimizer.step()

            # Print training set statistics
            batch_loss = torch.mean(loss)
            epoch_loss += loss
            log_interval = 400
            if batch_id % log_interval == 0 and batch_id > 0:    # print every 2000 mini-batches
                print('[epoch {}, batch id {}] batch loss{}'.format(
                    epoch, batch_id, batch_loss))
        epoch_loss = epoch_loss/num_epoch_utts
        print('[epoch {},  training loss:{},last training loss:{}'.format(
            epoch, epoch_loss, last_epoch_loss))

        # Adjust learning rate according cv loss
        cv_loss = cv(evalset, model)
        # print training set statistics
        print('[epoch {},  cv loss:{},last cv loss:{}'.format(
            epoch, cv_loss, last_cv_loss))
        # decay learning rate
        if cv_loss - last_cv_loss > 0:
            learning_rate = learning_rate/2
            print('adjust learning rate = {}'.format(learning_rate))
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        last_cv_loss = cv_loss
        last_epoch_loss = epoch_loss

        # Save model
        save_model_path = os.path.join(model_dir, 'epoch_{}.pt'.format(epoch))
        print('Checkpoint: save to checkpoint {}'.format(save_model_path))
        state_dict = model.state_dict()
        torch.save(
            {
                'model': state_dict,
                'epoch': epoch,
            }, save_model_path)

        # Stop condition
        if learning_rate < 1e-9:
            print('learning_rate too small = {}, stop training'.format(learning_rate))
            break

    # Save final_model
    save_model_path = os.path.join(model_dir, 'final.pt')
    torch.save(
        {
            'model': state_dict,
            'epoch': epoch,
        }, save_model_path)

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')

    parser.add_argument('--model_conf', required=True, help='model config')
    parser.add_argument('--train_data_dir', required=True,
                        help='kaldi data dir foramt')
    parser.add_argument('--cv_data_dir', required=True,
                        help='kaldi data dir foramt')
    #parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', help='output dir')

    args = parser.parse_args()

    # model_paras={
    #     'input_dim':120,
    #     'hidden_dim':640,
    #     'num_layers':4,
    #     'output_dim':74  #73 phone + 1 blank
    # }

    with open(args.model_conf) as fin:
        json_string = fin.read()
    model_paras = json.loads(json_string)

    epoch_num = 2

    feat_scp = os.path.join(args.train_data_dir, 'feats.sort.scp')
    label_file = os.path.join(args.train_data_dir, 'labels.scp')
    utt2spk = os.path.join(args.train_data_dir, 'utt2spk')
    cmvn_scp = os.path.join(args.train_data_dir, 'cmvn.scp')

    dataset = KaldiFeatureLabelReader(
        feat_scp, label_file, utt2spk, cmvn_scp, 8)

    feat_scp = os.path.join(args.cv_data_dir, 'feats.sort.scp')
    label_file = os.path.join(args.cv_data_dir, 'labels.scp')
    utt2spk = os.path.join(args.cv_data_dir, 'utt2spk')
    cmvn_scp = os.path.join(args.cv_data_dir, 'cmvn.scp')

    evalset = KaldiFeatureLabelReader(
        feat_scp, label_file, utt2spk, cmvn_scp, 8)

    # model_dir='/export/expts2/chaoyang/e2e/eesen/asr_egs/wsj/pytroch/model2/'
    train(dataset, evalset, epoch_num, model_paras,
          args.model_dir, last_model_path=False, use_cuda=True)
