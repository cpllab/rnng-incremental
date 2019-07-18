# Recurrent Neural Network Grammars

* Unkify input sentences: 

    `python get_raw.py train.02-21 eval_file.txt > unkified_eval_file.txt`

* Get word surprisal values:

    `build/nt-parser/nt-parser-gen --dynet-mem 2000  -x -T train_gen.oracle -v path/to/unkified_eval_file -f path/to/surprisals_output --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -m ntparse_gen_D0.3_2_256_256_16_256-pid20681.params`

    This will write the surprisal values to path/to/surprisals_output.

### References

[1] Dyer, C., Kuncoro, A., Ballesteros, M., & Smith, N. A. (2016). Recurrent Neural Network Grammars. In Proceedings of NAACL-HLT.

[2] Stern, M., Fried, D., & Klein, D. (2017). Effective Inference for Generative Neural Parsing. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing.

[3] Hale, J., Dyer, C., Kuncoro, A., & Brennan, J. (2018). Finding syntax in human encephalography with beam search. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).