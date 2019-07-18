#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "dynet/mp.h"
#include "rnnlm.h"
#include <boost/algorithm/string.hpp>

#include <iostream>
#include <fstream>
#include <vector>
/*
TODO:
- The shadow params in the trainers need to be shared.
*/

using namespace std;
using namespace dynet;
using namespace dynet::mp;
using namespace boost::interprocess;

typedef vector<int> Datum;

vector<Datum> ReadData(string filename) {
  vector<Datum> data;
  ifstream fs(filename);
  if (!fs.is_open()) {
    cerr << "ERROR: Unable to open " << filename << endl;
    exit(1);
  }
  string line;
  while (getline(fs, line)) {
    data.push_back(read_sentence(line, d));
  }
  return data;
}

template<class T, class D>
class Learner : public ILearner<D, dynet::real> {
public:
  explicit Learner(RNNLanguageModel<T>& rnnlm, unsigned data_size) : rnnlm(rnnlm) {}
  ~Learner() {}

  dynet::real LearnFromDatum(const D& datum, bool learn) {
    ComputationGraph cg;
    Expression loss_expr = rnnlm.BuildLMGraph(datum, cg);
    dynet::real loss = as_scalar(cg.forward(loss_expr));
    if (learn) {
      cg.backward(loss_expr);
    }
    return loss;
  }

  void SaveModel() {}

private:
  RNNLanguageModel<T>& rnnlm;
};

int main(int argc, char** argv) {
  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " cores corpus.txt dev.txt [iterations]" << endl;
    return 1;
  }
  srand(time(NULL));
  unsigned num_children = atoi(argv[1]);
  assert (num_children <= 64);
  vector<Datum> data = ReadData(argv[2]);
  vector<Datum> dev_data = ReadData(argv[3]);
  unsigned num_iterations = (argc >= 5) ? atoi(argv[4]) : UINT_MAX;
  unsigned dev_frequency = 5000;
  unsigned report_frequency = 10;

  dynet::initialize(argc, argv, true);

  ParameterCollection model;
  SimpleSGDTrainer trainer(model, 0.2);
  //AdagradTrainer trainer(model, 0.0);
  //AdamTrainer trainer(model, 0.0);

  RNNLanguageModel<LSTMBuilder> rnnlm(model);

  Learner<LSTMBuilder, Datum> learner(rnnlm, data.size());
  run_multi_process<Datum>(num_children, &learner, &trainer, data, dev_data, num_iterations, dev_frequency, report_frequency);
}
