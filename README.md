# ASR E2E acoustic model training with CTC in pytorch 

I am a newbie in pytorch. I create this repo to learn pythorch and 
E2E asr modeling.

For practical E2E ASR job, please refer to better tools such as espnet. 

## Current Feature
* Very simple pytorch implementation without optimization
* lstm+ctc, same config with eesen
* kaldi feature extraction (40 fbank+delta+cmvn)
* eesen TLG fst decoding 

## Usage
1. Download and install [Eesen](https://github.com/srvk/eesen)

2. in path.sh file, set EESEN_ROOT to eesen path.
```
export EESEN_ROOT='your/eesen/root/dir'
```

3. run run_ctc_phn.sh


## TODO
* [ ] CTC beamdecoder, phone/char/word LM
* [ ] Use RNN-t to involve LM.
* [ ] E2E structure 
* [ ] Try other Net structure.
