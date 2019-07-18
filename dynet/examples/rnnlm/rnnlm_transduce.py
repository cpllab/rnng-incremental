# a version rnnlm.py using the transduce() interface.
import dynet as dy
import time
import random

LAYERS = 2
INPUT_DIM = 50  #256
HIDDEN_DIM = 50  #1024
VOCAB_SIZE = 0

import argparse
import sys
import util
try:
    from itertools import izip as zip
except ImportError:
    pass

class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=dy.SimpleRNNBuilder):
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        self.lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
        self.R = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
        self.bias = model.add_parameters((VOCAB_SIZE))

    def BuildLMGraph(self, sent):
        dy.renew_cg()
        init_state = self.builder.initial_state()

        errs = [] # will hold expressions
        es=[]
        state = init_state
        inputs = [self.lookup[int(cw)] for cw in sent[:-1]]
        expected_outputs = [int(nw) for nw in sent[1:]]
        outputs = state.transduce(inputs)
        r_ts = ((self.bias + (self.R * y_t)) for y_t in outputs)
        errs = [dy.pickneglogsoftmax(r_t, eo) for r_t, eo in zip(r_ts, expected_outputs)]
        nerr = dy.esum(errs)
        return nerr

    def sample(self, first=1, nchars=0, stop=-1):
        # sampling must use the regular incremental interface.
        res = [first]
        dy.renew_cg()
        state = self.builder.initial_state()

        cw = first
        while True:
            x_t = self.lookup[cw]
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = self.bias + (self.R * y_t)
            ydist = dy.softmax(r_t)
            dist = ydist.vec_value()
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', help='Path to the corpus file.')
    args = parser.parse_args()

    train = util.CharsCorpusReader(args.corpus, begin="<s>")
    vocab = util.Vocab.from_corpus(train)
    
    VOCAB_SIZE = vocab.size()

    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)

    builder = dy.SimpleRNNBuilder
    # builder = dy.LSTMBuilder
    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=builder)

    train = list(train)

    chars = loss = 0.0
    for ITER in range(100):
        random.shuffle(train)
        for i,sent in enumerate(train):
            _start = time.time()
            if i % 50 == 0:
                trainer.status()
                if chars > 0: print(loss / chars,)
                for _ in range(1):
                    samp = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["\n"])
                    print("".join([vocab.i2w[c] for c in samp]).strip())
                loss = 0.0
                chars = 0.0
                
            chars += len(sent)-1
            isent = [vocab.w2i[w] for w in sent]
            errs = lm.BuildLMGraph(isent)
            loss += errs.scalar_value()
            errs.backward()
            trainer.update()
        print("ITER",ITER,loss)
        trainer.status()
