#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

stage=0
wsj0=/export/data/LDC/LDC93S6B
wsj1=/export/data/LDC/LDC94S13B

. utils/parse_options.sh

# add check for IRSTLM prune-lm
if ! prune-lm > /dev/null 2>&1; then
    echo "Error: prune-lm (part of IRSTLM) is not in path"
    echo "Make sure that you run tools/extras/install_irstlm.sh in the main Eesen directory;"
    echo " this is no longer installed by default."
    exit 1
fi


if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "             Data Preparation and FST Construction                 "
  echo =====================================================================
  # Use the same datap prepatation script from Kaldi
  #local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

  # Construct the phoneme-based lexicon from the CMU dict
  local/wsj_prepare_phn_dict.sh || exit 1;

  # Compile the lexicon and token FSTs
  utils/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

  # Compile the language-model FST and the final decoding graph TLG.fst
  local/wsj_decode_graph.sh data/lang_phn || exit 1;
fi
exit 1
if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================
  # Split the whole training data into training (95%) and cross-validation (5%) sets
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_si284 data/train_tr95 data/train_cv05 || exit 1

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  for set in train_tr95 train_cv05; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 14 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done

  for set in test_dev93 test_eval92; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 8 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                        Prepare Training data                      "
  echo =====================================================================
  # Label sequences; simply convert words into their label indices
  #for set in train_tr95 train_cv05; do
  for set in train_sample train_cv05; do
    python2 utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/$set/text "<UNK>" |tee data/$set/labels.scp |gzip -c - > data/$set/labels.gz
    utils/sort_feature_by_len.sh data/$set/feats.scp data/$set/feats.sort.scp 10
  done
fi
lstm_layer_num=4     # number of LSTM layers
lstm_cell_dim=640    # number of memory cells in every LSTM layer
dir=exp/train_phn_l${lstm_layer_num}_c${lstm_cell_dim}_pytorch  
if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                Pytorch Network Training                           "
  echo =====================================================================
  # Specify network structure and generate the network topology
  #input_feat_dim=120   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
  lstm_layer_num=4     # number of LSTM layers
  lstm_cell_dim=640    # number of memory cells in every LSTM layer
  #target_num=`cat data/local/dict_phn/units.txt | wc -l`; target_num=$[$target_num+1]; # the number of targets                    
  #epoch_num=10
  
  dir=exp/train_phn_l${lstm_layer_num}_c${lstm_cell_dim}_pytorch                                  
  mkdir -p $dir/model

  # Train the network with CTC. Refer to the script for details about the arguments
  python3 pytorch/train.py \
    --model_conf conf/model.json \
    --train_data_dir data/train_sample \
    --cv_data_dir data/train_cv05 \
    --model_dir $dir/model || exit 1;
fi

if [ $stage -le 4 ]; then
  # Send training/cv data ctc model and count greedy decoding result
  # Here we use a pre-count stats
  cp conf/label.counts $dir/priors.txt
  echo =====================================================================
  echo "                            Decoding                               "
  echo =====================================================================
  # Config for the basic decoding: --beam 30.0 --max-active 5000 --acoustic-scales "0.7 0.8 0.9"
  for lm_suffix in tgpr; do
    steps/decode_ctc_lat_pytorch.sh --cmd "$decode_cmd" --nj 10 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9 \
      data/lang_phn_test_${lm_suffix} $dir/model/final.pt data/test_dev93  $dir/test_dev93_${lm_suffix} || exit 1;
    
    steps/decode_ctc_lat_pytorch.sh --cmd "$decode_cmd" --nj 8 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9 \
      data/lang_phn_test_${lm_suffix} $dir/model/final.pt data/test_eval92  $dir/test_eval92_${lm_suffix} || exit 1;
  done
fi