# Word-synchronous Beam Search with Fast-tracking for Recurrent Neural Network Grammars

This repository implements word-synchronous beam search with fast-tracking (Stern et al. 2017; Hale et al., 2018) for Recurrent Neural Network Grammars (Dyer et al., 2016). 

## Set up the repository

Please download the latest version of [Dynet](https://github.com/clab/dynet) and put it in the root folder of this repository.

## Model training and evaluation

Please refers to the original [RNNG](https://github.com/clab/rnng) repository for instructions on model training and evaluation. Notice that since the code here uses [Dynet](https://github.com/clab/dynet), the flag `--cnn-mem` in the commands will need to be changed to `--dynet-mem`.

## Estimate word surprisals

### Unkify input sentences: 

    python get_raw.py train.02-21 eval_file.txt > unkified_eval_file.txt

### Get word surprisal values:

    build/nt-parser/nt-parser-gen --dynet-mem 2000  -x -T train_gen.oracle -v path/to/unkified_eval_file -f path/to/surprisals_output --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -m model_params_file

This will write the surprisal values to path/to/surprisals_output.

## Acknowledgement

We thank Miguel Ballesteros, Hui Wan, and Roger Levy for the help and advice with the implementation.

## References

[1] Dyer, C., Kuncoro, A., Ballesteros, M., & Smith, N. A. (2016). Recurrent Neural Network Grammars. In Proceedings of NAACL-HLT.

[2] Stern, M., Fried, D., & Klein, D. (2017). Effective Inference for Generative Neural Parsing. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing.

[3] Hale, J., Dyer, C., Kuncoro, A., & Brennan, J. (2018). Finding Syntax in Human Encephalography with Beam Search. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
