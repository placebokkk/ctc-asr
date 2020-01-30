#!/bin/bash
feat_scp=$1
sort_feat_scp=$2
min_len=$3
#$bin=/export/expts2/chaoyang/e2e/eesen/src/featbin/feat-to-len
feat-to-len scp:$feat_scp ark,t:- | awk '{print $2}' > len.tmp || exit 1;
paste -d " " $feat_scp len.tmp | sort -k3 -n - | awk -v m=$min_len '{ if ($3 >= m) {print $1 " " $2} }' > $sort_feat_scp || exit 1;
rm len.tmp