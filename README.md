# Recurrent Neural Network Grammars

* Unkify input sentences: 

    `python get_raw.py train.02-21 eval_file.txt > unkified_eval_file.txt`

* Get word surprisal values:

    `build/nt-parser/nt-parser-gen --dynet-mem 2000  -x -T train_gen.oracle -v path/to/unkified_eval_file -f path/to/surprisals_output --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -m ntparse_gen_D0.3_2_256_256_16_256-pid20681.params`

    This will write the surprisal values to path/to/surprisals_output.