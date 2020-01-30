#!/bin/bash

# Apache 2.0

# Decode the CTC-trained model by generating lattices.


## Begin configuration section
nj=16
cmd=run.pl

acwt=0.9

min_active=200
max_active=7000 # max-active
beam=15.0       # beam used
lattice_beam=8.0
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes

skip_scoring=false # whether to skip WER scoring
scoring_opts="--min-acwt 5 --max-acwt 10 --acwt-factor 0.1"
score_with_conf=false


echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: steps/decode_ctc.sh [options] <graph-dir> <model-path> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_ctc.sh data/lang exp/train_l4_c320/model/final.pt data/test exp/train_l4_c320/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt                                   # default 0.9, the acoustic scale to be used"
   exit 1;
fi

graphdir=$1
model=$2
data=$3
dir=`echo $4 | sed 's:/$::g'` # remove any trailing slash.
srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
echo $graphdir $model $data  $dir

# Check if necessary files exist.
for f in $graphdir/TLG.fst; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

$cmd JOB=1:$nj $dir/log/sort_feature.JOB.log \
utils/sort_feature_by_len.sh $data/split$nj/JOB/feats.scp $data/split$nj/JOB/feats.sort.scp 1

# Do this in order
for JOB in `seq $nj`;do 
   python3 pytorch/net-output-extract.py \
  --model_conf conf/model.json \
  --prior_file $srcdir/priors.txt \
  --model_file $model \
  --data_dir $data/split$nj/$JOB/ \
  --out_file $srcdir/out_prob.$JOB.txt
done

# Decode for each of the acoustic scales
$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  latgen-faster  --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$lattice_beam \
  --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $graphdir/TLG.fst ark,t:$srcdir/out_prob.JOB.txt "ark:|gzip -c > $dir/lat.JOB.gz" || \
exit 1;

# Scoring
if ! $skip_scoring ; then
  if [ -f $data/stm ]; then # use sclite scoring.
    if $score_with_conf ; then
      [ ! -x local/score_sclite_conf.sh ] && echo "Not scoring because local/score_sclite_conf.sh does not exist or not executable." && exit 1;
      local/score_sclite_conf.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
    else
      [ ! -x local/score_sclite.sh ] && echo "Not scoring because local/score_sclite.sh does not exist or not executable." && exit 1;
      local/score_sclite.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
    fi
  else
    [ ! -x local/score.sh ] && echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
  fi
fi

exit 0;
