# Word-synchronous Beam Search with Fast-tracking for Recurrent Neural Network Grammars

This repository implements word-synchronous beam search with fast-tracking (Stern et al. 2017; Hale et al., 2018) for Recurrent Neural Network Grammars (Dyer et al., 2016), as well as an ablated RNNG model referred as "ActionLSTM" in Wilcox et al. (2019). The ablated model is implemented in `nt-parser/nt-parser-gen-action-only.cc`,  which only uses the encoder of the action sequence, similar to Choe and Charniak (2016).

## Set up the repository

Please put the latest version of [Dynet](https://github.com/clab/dynet) in the root folder of this repository, and refer to the original [RNNG](https://github.com/clab/rnng) repository for instructions on building the repository, preparing oracle files, and training and evaluating models. The oracle files for training and evaluating the generative RNNG (`nt-parser-gen`) also work for `nt-parser-gen-action-only`. Notice that as the code here uses [Dynet](https://github.com/clab/dynet), the flag `--cnn-mem` in the commands will need to be changed to `--dynet-mem`.

## Estimate word surprisals

### Prepare the vocabulary file:

    python get_dictionary.py train.02-21 > train_vocab.txt

### Unkify input sentences: 

Assuming a list of tokenized sentences as the text file `eval_file.txt`, with one sentence per line:

    python get_unkified_input.py train_vocab.txt eval_file.txt > unkified_eval_file.txt

### Get word surprisal values:

    build/nt-parser/nt-parser-gen --dynet-mem 2000  -x -T train_gen.oracle -v path/to/unkified_eval_file -f path/to/surprisals_output --clusters path/to/clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -m model_params_file

This will write the surprisal values to path/to/surprisals_output.

## Acknowledgement

We thank Miguel Ballesteros, Hui Wan, and Roger Levy for the help and advice with the implementation.

## References

[1] Choe, D., & Charniak, E. (2016). Parsing as Language Modeling. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing.

[2] Dyer, C., Kuncoro, A., Ballesteros, M., & Smith, N. A. (2016, June). Recurrent Neural Network Grammars. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.

[3] Hale, J., Dyer, C., Kuncoro, A., & Brennan, J. (2018). Finding Syntax in Human Encephalography with Beam Search. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

[4] Stern, M., Fried, D., & Klein, D. (2017). Effective Inference for Generative Neural Parsing. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing.

[5] Wilcox, E., Qian, P., Futrell, R., Ballesteros, M., & Levy, R. (2019, June). Structural Supervision Improves Learning of Non-Local Grammatical Dependencies. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers).
