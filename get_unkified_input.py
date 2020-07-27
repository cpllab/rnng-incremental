import sys
import get_dictionary
import get_oracle_gen


def main():
    if len(sys.argv) != 3:
        raise NotImplementedError('Program only takes two arguments: vocab file and eval file')
    vocab_file = open(sys.argv[1], 'r')
    lines = vocab_file.readlines()
    vocab_file.close()
    words_set = set([line.strip() for line in lines])

    eval_file = open(sys.argv[2], 'r')
    eval_lines = eval_file.readlines()
    eval_file.close()

    for line in eval_lines:
        tokens = line.strip().split()
        unkified = get_oracle_gen.unkify(tokens, words_set)    
        print ' '.join(unkified)
    

if __name__ == "__main__":
    main()
