#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/globals.h"
#include "dynet/io.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;

unsigned INPUT_DIM = 36;
unsigned OUTPUT_DIM = 36;
unsigned VOCAB_SIZE = 0;
unsigned LABEL_SIZE = 0;
float pdropout = 0.5;

dynet::Dict d;
dynet::Dict ld;
int kSOS;
int kEOS;

struct NeuralBagOfWords {
  LookupParameter p_w;
  Parameter p_c2h;
  Parameter p_hbias;
  Parameter p_h2o;
  Parameter p_obias;

  explicit NeuralBagOfWords(ParameterCollection& m) :
      p_w(m.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_c2h(m.add_parameters({OUTPUT_DIM, INPUT_DIM})),
      p_hbias(m.add_parameters({OUTPUT_DIM})),
      p_h2o(m.add_parameters({LABEL_SIZE, OUTPUT_DIM})),
      p_obias(m.add_parameters({LABEL_SIZE})) {}

  Expression BuildClassifier(const vector<int>& x, ComputationGraph& cg) {
    Expression c2h = parameter(cg, p_c2h);
    Expression hbias = parameter(cg, p_hbias);
    Expression h2o = parameter(cg, p_h2o);
    Expression obias = parameter(cg, p_obias);

    vector<Expression> vx(x.size());
    for (unsigned i = 0; i < x.size(); ++i)
      vx[i] = lookup(cg, p_w, x[i]);
    Expression c = sum(vx);
    Expression h = rectify(c2h * c / x.size() + hbias);
    Expression y_pred = obias + h2o * h;
    return y_pred;
  }
};

struct ConvLayer {
  // in_rows = rows per word in input matrix
  // k_fold_rows = 1 no folding, 2 fold two rows together, 3 ...
  // filter_width = length of filter (columns)
  // in_nfmaps = number of feature maps in input
  // out_nfmaps = number of feature maps in output
  ConvLayer(ParameterCollection& m, int in_rows, int k_fold_rows, int filter_width, int in_nfmaps, int out_nfmaps) :
      p_filts(in_nfmaps),
      p_fbias(in_nfmaps),
      k_fold_rows(k_fold_rows) {
    if (k_fold_rows < 1 || ((in_rows / k_fold_rows) * k_fold_rows != in_rows)) {
      cerr << "Bad k_fold_rows=" << k_fold_rows << endl;
      abort();
    }
    for (int i = 0; i < in_nfmaps; ++i) {
      p_filts[i].resize(out_nfmaps);
      p_fbias[i].resize(out_nfmaps);
      for (int j = 0; j < out_nfmaps; ++j) {
        p_filts[i][j] = m.add_parameters({(unsigned)in_rows, (unsigned)filter_width}, 0.01);
        p_fbias[i][j] = m.add_parameters({(unsigned)in_rows}, 0.05);
      }
    }
    //for (int j = 0; j < out_nfmaps; ++j)
      //p_fbias[j] = m.add_parameters({in_rows});
  }

  vector<Expression> apply(ComputationGraph& cg, const vector<Expression>& inlayer, int k_out) const {
    const unsigned out_nfmaps = p_filts.front().size();
    const unsigned in_nfmaps = p_filts.size();
    if (in_nfmaps != inlayer.size()) {
      cerr << "Mismatched number of input features (" << inlayer.size() << "), expected " << in_nfmaps << endl;
      abort();
    }
    vector<Expression> r(out_nfmaps);

    vector<Expression> tmp(in_nfmaps);
    for (unsigned fj = 0; fj < out_nfmaps; ++fj) {
      for (unsigned fi = 0; fi < in_nfmaps; ++fi) {
        Expression t = conv1d_wide(inlayer[fi], parameter(cg, p_filts[fi][fj]));
        t = colwise_add(t, parameter(cg, p_fbias[fi][fj]));
        tmp[fi] = t;
      }
      Expression s = sum(tmp);
      if (k_fold_rows > 1)
        s = fold_rows(s, k_fold_rows);
      s = kmax_pooling(s, k_out);
      r[fj] = rectify(s);
    }
    return r;
  }
  vector<vector<Parameter>> p_filts; // [feature map index from][feature map index to]
  vector<vector<Parameter>> p_fbias; // [feature map index from][feature map index to]
  int k_fold_rows;
};

struct ConvNet {
  LookupParameter p_w;
  ConvLayer cl1;
  ConvLayer cl2;
  Parameter p_t2o;
  Parameter p_obias;

  explicit ConvNet(ParameterCollection& m) :
      p_w(m.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
  //ConvLayer(ParameterCollection& m, int in_rows, int k_fold_rows, int filter_width, int in_nfmaps, int out_nfmaps) :
      cl1(m, INPUT_DIM, 2,  10, 1, 6),
      cl2(m, INPUT_DIM/2, 2, 6, 6, 14),
      p_t2o(m.add_parameters({LABEL_SIZE, 14 * (INPUT_DIM / 4) * 5})),
      p_obias(m.add_parameters({LABEL_SIZE})) {
  }

  Expression BuildClassifier(const vector<int>& x, ComputationGraph& cg, bool for_training) {
    Expression t2o = parameter(cg, p_t2o);
    Expression obias = parameter(cg, p_obias);
    int k_2 = 5;
    int len = x.size();
    int k_1 = max(k_2, len / 2);
    vector<Expression> vx(x.size());
    for (unsigned i = 0; i < x.size(); ++i)
      vx[i] = lookup(cg, p_w, x[i]);
    Expression s = concatenate_cols(vx);
    
    vector<Expression> l0(1, s);
    vector<Expression> l1 = cl1.apply(cg, l0, k_1);
    vector<Expression> l2 = cl2.apply(cg, l1, k_2);
    for(auto& fm : l2)
      fm = reshape(fm, {k_2 * INPUT_DIM / 4});
    Expression t = concatenate(l2);
    if (for_training)
      t = dropout(t, pdropout);
    Expression r = t2o * t + obias;
    return r;
  }
};

bool IsCurrentPredictionCorrection(Expression y_pred, int y_true) {
  ComputationGraph& cg = *y_pred.pg;
  auto v = as_vector(cg.incremental_forward(y_pred));
  assert(v.size() > 1);
  int besti = 0;
  float best = v[0];
  for (unsigned i = 1; i < v.size(); ++i)
    if (v[i] > best) { best = v[i]; besti = i; }
  return (besti == y_true);
}

Expression CrossEntropyLoss(const Expression& y_pred, int y_true) {
  Expression lp = log_softmax(y_pred);
  Expression nll = -pick(lp, y_true);
  return nll;
}

Expression HingeLoss(const Expression& y_pred, int y_true) {
  Expression hl = hinge(y_pred, y_true, 10.0f);
  return hl;
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.file]\n";
    return 1;
  }
  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");
  vector<pair<vector<int>,int>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      vector<int> x,y;
      read_sentence_pair(line, x, d, y, ld);
      if (x.size() == 0 || y.size() != 1) { cerr << line << endl; abort(); }
      training.push_back(make_pair(x,y[0]));
      ttoks += x.size();
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
    cerr << "Labels: " << ld.size() << endl;
  }
  LABEL_SIZE = ld.size();
  //d.freeze(); // no new word types allowed
  ld.freeze(); // no new tag types allowed

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      vector<int> x,y;
      read_sentence_pair(line, x, d, y, ld);
      assert(y.size() == 1);
      dev.push_back(make_pair(x,y[0]));
      dtoks += x.size();
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  VOCAB_SIZE = d.size();
  ostringstream os;
  os << "textcat"
     << '_' << INPUT_DIM
     << '_' << OUTPUT_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  ParameterCollection model;
  //trainer = new MomentumSGDTrainer(model);
  std::unique_ptr<Trainer> trainer(new AdagradTrainer(model));
  //trainer = new SimpleSGDTrainer(model);

  //NeuralBagOfWords nbow(model);
  ConvNet nbow(model);

  if (argc == 4) {
    TextFileLoader loader(argv[3]);
    loader.populate(model);
  }

  unsigned report_every_i = min(100, int(training.size()));
  unsigned dev_every_i_reports = 25;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned ttags = 0;
    unsigned correct = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
        si = 0;
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }

      // build graph for this instance
      ComputationGraph cg;
      auto& sentx_y = training[order[si]];
      const auto& x = sentx_y.first;
      const int y = sentx_y.second;
      ++si;
      //cerr << "LINE: " << order[si] << endl;
      Expression y_pred = nbow.BuildClassifier(x, cg, true);
      //Expression loss_expr = CrossEntropyLoss(y_pred, y);
      Expression loss_expr = HingeLoss(y_pred, y);
      loss += as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      trainer->update();
      ++lines;
      ++ttags;
    }
    trainer->status();
    cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / (double)ttags) << ") ";
    model.project_weights();

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dtags = 0;
      unsigned dcorr = 0;
      for (auto& sent : dev) {
        const auto& x = sent.first;
        const int y = sent.second;
        nbow.p_t2o.get_storage().scale_parameters(pdropout);
        ComputationGraph cg;
        Expression y_pred = nbow.BuildClassifier(x, cg, false);
        if (IsCurrentPredictionCorrection(y_pred, y)) dcorr++;
        //Expression loss_expr = CrossEntropyLoss(y_pred, y);
        Expression loss_expr = HingeLoss(y_pred, y);
        //cerr << "DEVLINE: " << dtags << endl;
        dloss += as_scalar(cg.incremental_forward(loss_expr));
        nbow.p_t2o.get_storage().scale_parameters(1.f/pdropout);
        dtags++;
      }
      if (dloss < best) {
        best = dloss;
        TextFileSaver saver("textcat.model");
        saver.save(model);
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << " acc=" << (dcorr / (double)dtags) << ' ';
    }
  }
}
