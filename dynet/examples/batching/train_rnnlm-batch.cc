/**
 * Train a RNN language model in a batched manner
 *
 * This provide an example of usage of the rnnlm-batch.h model
 */
#include "rnnlm-batch.h"
#include "dynet/io.h"
#include "cl-args.h"
#include <cassert>

using namespace std;
using namespace dynet;

// Sort in descending order of length
struct CompareLen {
  bool operator()(const std::vector<int>& first, const std::vector<int>& second) {
    return first.size() > second.size();
  }
};

int main(int argc, char** argv) {
  // Fetch dynet params ----------------------------------------------------------------------------
  auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dyparams);
  // Fetch program specific parameters (see ../utils/cl-args.h) ------------------------------------
  Params params;

  get_args(argc, argv, params, TRAIN);

  // Load datasets ---------------------------------------------------------------------------------

  // Dictionary
  Dict d;

  // Start and end of sentence tokens
  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");

  // Datasets
  vector<vector<int>> training, dev;

  // Read training data and fill dictionary
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << params.train_file << "...\n";
  {
    ifstream in(params.train_file);
    assert(in);
    while (getline(in, line)) {
      ++tlc;
      training.push_back(read_sentence(line, d));
      ttoks += training.back().size();
      if (training.back().front() != kSOS && training.back().back() != kEOS) {
        cerr << "Training sentence in " << params.train_file << ":" << tlc
             << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  // Sort the training sentences in descending order of length
  CompareLen comp;
  sort(training.begin(), training.end(), comp);
  // Pad the sentences in the same batch with EOS so they are the same length
  // This modifies the training objective a bit by making it necessary to
  // predict EOS multiple times, but it's easy and not so harmful
  for (size_t i = 0; i < training.size(); i += params.BATCH_SIZE)
    for (size_t j = 1; j < params.BATCH_SIZE; ++j)
      while (training[i + j].size() < training[i].size())
        training[i + j].push_back(kEOS);
  // Freeze dictionary
  d.freeze(); // no new word types allowed
  d.set_unk("UNK");
  INPUT_VOCAB_SIZE = d.size();
  OUTPUT_VOCAB_SIZE = d.size();

  // Read validation dataset
  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << params.dev_file << "...\n";
  {
    ifstream in(params.dev_file);
    assert(in);
    while (getline(in, line)) {
      ++dlc;
      dev.push_back(read_sentence(line, d));
      dtoks += dev.back().size();
      if (dev.back().front() != kSOS && dev.back().back() != kEOS) {
        cerr << "Dev sentence in " << params.dev_file << ":" << dlc
             << " didn't start or end with <s>, </s>\n";
        cerr << d.convert(dev.back().front())  << ":"
             << d.convert(dev.back().back()) << " \n";
        cerr << kSOS << ":" << kEOS << "\n";
        abort();
      }
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  // Sort the dev sentences in descending order of length (for minibatching)
  sort(dev.begin(), dev.end(), comp);
  // Pad the sentences in the same batch with EOS so they are the same length
  // This modifies the dev objective a bit by making it necessary to
  // predict EOS multiple times, but it's easy and not so harmful
  for (size_t i = 0; i < dev.size(); i += params.DEV_BATCH_SIZE)
    for (size_t j = 1; j < params.DEV_BATCH_SIZE; ++j)
      while (dev[i + j].size() < dev[i].size())
        dev[i + j].push_back(kEOS);

  // Model name (for saving) -----------------------------------------------------------------------
  ostringstream os;
  // Store a bunch of information in the model name
  os << params.exp_name
     << "_" << "rnnlm"
     << '_' << params.LAYERS
     << '_' << params.INPUT_DIM
     << '_' << params.HIDDEN_DIM
     << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;

  // Initialize model and trainer ------------------------------------------------------------------
  ParameterCollection model;
  // Use Adam optimizer
  std::unique_ptr<Trainer> trainer(new AdamTrainer(model, 0.001, 0.9, 0.999, 1e-8));
  trainer->clip_threshold *= params.BATCH_SIZE;

  // Create model
  RNNBatchLanguageModel<LSTMBuilder> lm(model,
                                   params.LAYERS,
                                   params.INPUT_DIM,
                                   params.HIDDEN_DIM,
                                   INPUT_VOCAB_SIZE);

  // Load preexisting weights (if provided)
  if (params.model_file != "") {
    TextFileLoader loader(params.model_file);
    loader.populate(model);
  }

  // Initialize variables for training -------------------------------------------------------------
  // Best dev score
  double best = 9e+99;

  // Number of batches in training set
  unsigned num_batches = training.size()  / params.BATCH_SIZE - 1;
  // Number of batches in validation set
  unsigned num_dev_batches = dev.size() / params.DEV_BATCH_SIZE - 1;

  // Number of sentences to sample each epoch (for visualization)
  unsigned size_samples = 200;

  // Random indexing
  unsigned si;
  vector<unsigned> order(num_batches);
  for (unsigned i = 0; i < num_batches; ++i) order[i] = i;

  unsigned epoch = 0;
  // Run for the given number of epochs (or indefinitely if params.NUM_EPOCHS is negative)
  while (static_cast<int>(epoch) < params.NUM_EPOCHS || params.NUM_EPOCHS < 0) {
    // Reshuffle the dataset
    cerr << "**SHUFFLE\n";
    random_shuffle(order.begin(), order.end());
    // Initialize loss and number of chars(/word) (for loss per char/word)
    double loss = 0;
    unsigned tokens = 0;
    // Start timer
    std::unique_ptr<Timer> iteration(new Timer("completed in"));

    for (si = 0; si < num_batches; ++si) {
      // build graph for this instance
      ComputationGraph cg;
      // Compute batch start id and size
      int id = order[si] * params.BATCH_SIZE;
      unsigned bsize = std::min((unsigned)training.size() - id, params.BATCH_SIZE);
      // Get negative log likelihood on batch
      Expression loss_expr = lm.getNegLogProb(training, id, bsize, tokens, cg);
      // Get scalar error for monitoring
      loss += as_scalar(cg.forward(loss_expr));
      // Compute gradient with backward pass
      cg.backward(loss_expr);
      // Update parameters
      trainer->update();
      // Print progress every tenth of the dataset
      if ((si + 1) % (num_batches / 10) == 0 || si == num_batches - 1) {
        // Print informations
        trainer->status();
        cerr << " E = " << (loss / tokens) << " ppl=" << exp(loss / tokens) << ' ';
        // Reinitialize timer
        iteration.reset(new Timer("completed in"));
        // Reinitialize loss
        loss = 0;
        tokens = 0;
      }
    }


    // Show score on dev data
    if (si == num_batches) {
      double dloss = 0;
      unsigned dtokens = 0;
      for (unsigned i = 0; i < num_dev_batches; ++i) {
        // build graph for this instance
        ComputationGraph cg;
        // Compute batch start id and size
        unsigned id = i * params.DEV_BATCH_SIZE;
        unsigned bsize = std::min((unsigned)dev.size() - id, params.DEV_BATCH_SIZE); // Batch size
        // Get negative log likelihood on batch
        Expression loss_expr = lm.getNegLogProb(dev, id, bsize, dtokens, cg);
        // Add loss
        dloss += as_scalar(cg.forward(loss_expr));
      }
      // If the dev loss is lower than the previous ones, save the model
      if (dloss < best) {
        best = dloss;
        TextFileSaver saver("rnnlm-batch.model");
        saver.save(model);
      }
      // Print informations
      cerr << "\n***DEV [epoch=" << (epoch)
           << "] E = " << (dloss / dtokens)
           << " ppl=" << exp(dloss / dtokens) << ' ';
      // Reinitialize timer
      iteration.reset(new Timer("completed in"));
    }

    // Sample some examples because it's cool (also helps debugging)
    cout << "---------------------------------------------" << endl;
    lm.RandomSample(d, size_samples);
    cout << "---------------------------------------------" << endl;

    // Increment epoch
    ++epoch;
  }
}
