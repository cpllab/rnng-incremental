/**This file implements word-synchronous beam search with fast-tracking and particle
 * filtering to estimate word surprisals*/

#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/functional/hash.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "dynet/dict.h"
#include "dynet/cfsm-builder.h"
#include "dynet/io.h"

#include "nt-parser/oracle.h"
#include "nt-parser/pretrained.h"
#include "nt-parser/compressed-fstream.h"
#include "nt-parser/eval.h"

// dictionaries
dynet::Dict termdict, ntermdict, adict, posdict;

unsigned debug_level = 5;
volatile bool requested_stop = false;
unsigned kSOS = 0;
unsigned IMPLICIT_REDUCE_AFTER_SHIFT = 0;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;

unsigned BEAM_SIZE = 100;
unsigned FASTTRACK_BEAM_SIZE = 5;
unsigned WORD_BEAM_SIZE = 10;

unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
float DROPOUT = 0.0f;
float LEARNING_RATE = 0.1;
unsigned NUM_PARTICLES = 100;
std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X

using namespace dynet;
using namespace std;
namespace po = boost::program_options;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;

ClassFactoredSoftmaxBuilder *cfsm = nullptr;

/**Get, parse, and return the command-line arguments*/
void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("explicit_terminal_reduce,x", "[not recommended] If set, the parser must explicitly process a REDUCE operation to complete a preterminal constituent")
        ("dropout,D", po::value<float>(), "Use dropout")
        ("clusters,c", po::value<string>(), "Clusters word clusters file")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data,p", po::value<string>(), "Test corpus")
        ("eval_data,v", po::value<string>(), "Evaluation corpus")
        ("file_path,f", po::value<string>(), "File path for saving evaluation data")
        ("eta_decay,e", po::value<float>(), "Start decaying eta after this many epochs")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("beam_size", po::value<unsigned>()->default_value(100), "beam size")
        ("np", po::value<unsigned>()->default_value(100), "number of particle filters for particle filtering")
        ("beam", po::value<bool>()->default_value(false), "true if running beam search")
        ("fasttrack_beam_size", po::value<unsigned>()->default_value(5), "fast track beam size")
        ("word_beam_size", po::value<unsigned>()->default_value(10), "word beam size")
        ("lr", po::value<float>()->default_value(0.1), "Learning rate")  
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --training_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

/**The structure that stores the states and parsing functions (greedy, beam, particle) of our parser*/
struct ParserBuilder {
  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder term_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder action_lstm;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;
  LookupParameter p_w; // word embeddings
  LookupParameter p_t; // pretrained word embeddings (not updated)
  LookupParameter p_nt; // nonterminal embeddings
  LookupParameter p_ntup; // nonterminal embeddings when used in a composed representation
  LookupParameter p_a; // input action embeddings
  Parameter p_pbias; // parser state bias
  Parameter p_A; // action lstm to parser state
  Parameter p_S; // stack lstm to parser state
  Parameter p_T; // term lstm to parser state
  Parameter p_cbias; // composition function bias
  Parameter p_p2a;   // parser state to action
  Parameter p_action_start;  // action bias
  Parameter p_abias;  // action bias
  Parameter p_stack_guard;  // end of stack
  Parameter p_cW;

  /**constructor function for the ParserBuilder structure*/
  explicit ParserBuilder(ParameterCollection* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, *model),
      term_lstm(LAYERS, INPUT_DIM, HIDDEN_DIM, *model),  // sequence of generated terminals
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, *model),
      const_lstm_fwd(1, LSTM_INPUT_DIM, LSTM_INPUT_DIM, *model), // used to compose children of a node into a representation of the node
      const_lstm_rev(1, LSTM_INPUT_DIM, LSTM_INPUT_DIM, *model), // used to compose children of a node into a representation of the node
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_t(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_nt(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_ntup(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_T(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM})),
      p_abias(model->add_parameters({ACTION_SIZE})),
      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM})),
      p_cW(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM * 2})) {
    if (pretrained.size() > 0) {
      cerr << "Pretrained embeddings not implemented\n";
      abort();
    }
  }

/**Helper function used by log_prob_parser to determine if the action is allowed or forbidden*/
static bool IsActionForbidden_Generative(const string& a, char prev_a, unsigned tsize, unsigned ssize, unsigned nopen_parens) {
  bool is_shift = (a[0] == 'S' && a[1]=='H');
  bool is_reduce = (a[0] == 'R' && a[1]=='E');
  bool is_nt = (a[0] == 'N');
  assert(is_shift || is_reduce || is_nt);
  static const unsigned MAX_OPEN_NTS = 100;
  if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
  if (ssize == 1) {
    if (!is_nt) return true;
    return false;
  }
  // you can't reduce after an NT action
  if (is_reduce && prev_a == 'N') return true;
  return false;
}

/**Returns parse actions for input sentence (in training just returns the reference)
this lets us use pretrained embeddings, when available, for words that were OOV in the
parser training data if sent is empty, generate a sentence*/
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     Expression& loss,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
                     bool is_evaluation) {
    vector<unsigned> results;
    vector<string> stack_content;
    stack_content.push_back("ROOT_GUARD");
    const bool sample = sent.size() == 0;
    const bool build_training_graph = correct_actions.size() > 0;
    assert(sample || build_training_graph);
    bool apply_dropout = (DROPOUT && !is_evaluation);
    if (sample) apply_dropout = false;

    if (apply_dropout) {
      stack_lstm.set_dropout(DROPOUT);
      term_lstm.set_dropout(DROPOUT);
      action_lstm.set_dropout(DROPOUT);
      const_lstm_fwd.set_dropout(DROPOUT);
      const_lstm_rev.set_dropout(DROPOUT);
    } else {
      stack_lstm.disable_dropout();
      term_lstm.disable_dropout();
      action_lstm.disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
    }
    term_lstm.new_graph(*hg);
    stack_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    cfsm->new_graph(*hg);
    term_lstm.start_new_sequence();
    stack_lstm.start_new_sequence();
    action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression A = parameter(*hg, p_A);
    Expression T = parameter(*hg, p_T);
    //Expression pbias2 = parameter(*hg, p_pbias2);
    //Expression S2 = parameter(*hg, p_S2);
    //Expression A2 = parameter(*hg, p_A2);

    //Expression ib = parameter(*hg, p_ib);
    Expression cbias = parameter(*hg, p_cbias);
    //Expression w2l = parameter(*hg, p_w2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);

    action_lstm.add_input(action_start);

    vector<Expression> terms(1, lookup(*hg, p_w, kSOS));
    term_lstm.add_input(terms.back());

    vector<Expression> stack;  // variables representing subtree embeddings
    stack.push_back(parameter(*hg, p_stack_guard));
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    vector<Expression> log_probs;
    string rootword;
    unsigned action_count = 0;  // incremented at each prediction
    unsigned nt_count = 0; // number of times an NT has been introduced
    vector<unsigned> current_valid_actions;
    int nopen_parens = 0;
    char prev_a = '0';
    unsigned termc = 0;
    while(stack.size() > 2 || termc == 0) {
      assert (stack.size() == stack_content.size());
      // get list of possible actions for the current parser state
      current_valid_actions.clear();
      for (auto a: possible_actions) {
        if (IsActionForbidden_Generative(adict.convert(a), prev_a, terms.size(), stack.size(), nopen_parens))
          continue;
        current_valid_actions.push_back(a);
      }
      //cerr << "valid actions = " << current_valid_actions.size() << endl;

      //onerep
      Expression stack_summary = stack_lstm.back();
      Expression action_summary = action_lstm.back();
      Expression term_summary = term_lstm.back();
      if (apply_dropout) {
        stack_summary = dropout(stack_summary, DROPOUT);
        action_summary = dropout(action_summary, DROPOUT);
        term_summary = dropout(term_summary, DROPOUT);
      }
      Expression p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
      Expression nlp_t = rectify(p_t);
      //tworep*
      //Expression p_t = affine_transform({pbias, S, stack_lstm.back(), A, action_lstm.back()});
      //Expression nlp_t = rectify(p_t);
      //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});

      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      unsigned action = 0;
      if (sample) {
        auto dist = as_vector(hg->incremental_forward(adiste));
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(dist[current_valid_actions[w]]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        action = current_valid_actions[w];
        const string& a = adict.convert(action);
        if (a[0] == 'R') cerr << ")";
        if (a[0] == 'N') {
          int nt = action2NTindex[action];
          cerr << " (" << ntermdict.convert(nt);
        }
      } else {
        if (action_count >= correct_actions.size()) {
          cerr << "Correct action list exhausted, but not in final parser state.\n";
          abort();
        }
        action = correct_actions[action_count];
        //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.Convert(i) << ':' << adist[i]; }
        //cerr << endl;
        ++action_count;
        log_probs.push_back(pick(adiste, action));
      }
      results.push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

      // do action
      const string& actionString=adict.convert(action);
      //cerr << "ACT: " << actionString << endl;
      const char ac = actionString[0];
      const char ac2 = actionString[1];
      prev_a = ac;

      if (ac =='S' && ac2=='H') {  // SHIFT
        unsigned wordid = 0;
        //tworep
        //Expression p_t = affine_transform({pbias2, S2, stack_lstm.back(), A2, action_lstm.back(), T, term_lstm.back()});
        //Expression nlp_t = rectify(p_t);
        //tworep-oneact:
        //Expression p_t = affine_transform({pbias2, S2, stack_lstm.back(), T, term_lstm.back()});
        //Expression nlp_t = rectify(p_t);
        if (sample) {
          wordid = cfsm->sample(nlp_t);
          cerr << " " << termdict.convert(wordid);
        } else {
          assert(termc < sent.size());
          wordid = sent.raw[termc];
          log_probs.push_back(-cfsm->neg_log_softmax(nlp_t, wordid));
        }
        assert (wordid != 0);
        stack_content.push_back(termdict.convert(wordid)); //add the string of the word to the stack
        ++termc;
        Expression word = lookup(*hg, p_w, wordid);
        terms.push_back(word);
        term_lstm.add_input(word);
        stack.push_back(word);
        stack_lstm.add_input(word);
        is_open_paren.push_back(-1);
      } else if (ac == 'N') { // NT
        ++nopen_parens;
        auto it = action2NTindex.find(action);
        assert(it != action2NTindex.end());
        int nt_index = it->second;
        nt_count++;
        stack_content.push_back(ntermdict.convert(nt_index));
        Expression nt_embedding = lookup(*hg, p_nt, nt_index);
        stack.push_back(nt_embedding);
        stack_lstm.add_input(nt_embedding);
        is_open_paren.push_back(nt_index);
      } else { // REDUCE
        --nopen_parens;
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
        assert(stack_content.size() > 2 && stack.size() == stack_content.size());
        // find what paren we are closing
        int i = is_open_paren.size() - 1; //get the last thing on the stack
        while(is_open_paren[i] < 0) { --i; assert(i >= 0); } //iteratively decide whether or not it's a non-terminal
        Expression nonterminal = lookup(*hg, p_ntup, is_open_paren[i]);
        int nchildren = is_open_paren.size() - i - 1;
        assert(nchildren > 0);
        //cerr << "  number of children to reduce: " << nchildren << endl;
        vector<Expression> children(nchildren);
        const_lstm_fwd.start_new_sequence();
        const_lstm_rev.start_new_sequence();

        // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
        // TO BE COMPOSED INTO A TREE EMBEDDING
        string curr_word;
        //cerr << "--------------------------------" << endl;
        //cerr << "Now printing the children" << endl;
        //cerr << "--------------------------------" << endl;
        for (i = 0; i < nchildren; ++i) {
          assert (stack_content.size() == stack.size());
          children[i] = stack.back();
          stack.pop_back();
          stack_lstm.rewind_one_step();
          is_open_paren.pop_back();
          curr_word = stack_content.back();
          //cerr << "At the back of the stack (supposed to be one of the children): " << curr_word << endl;
          stack_content.pop_back();
        }
        assert (stack_content.size() == stack.size());
        //cerr << "Doing REDUCE operation" << endl;
        is_open_paren.pop_back(); // nt symbol
        stack.pop_back(); // nonterminal dummy
        stack_lstm.rewind_one_step(); // nt symbol
        curr_word = stack_content.back();
        //cerr << "--------------------------------" << endl;
        //cerr << "At the back of the stack (supposed to be the non-terminal symbol) : " << curr_word << endl;
        stack_content.pop_back();
        assert (stack.size() == stack_content.size());
        //cerr << "Done reducing" << endl;

        // BUILD TREE EMBEDDING USING BIDIR LSTM
        const_lstm_fwd.add_input(nonterminal);
        const_lstm_rev.add_input(nonterminal);
        for (i = 0; i < nchildren; ++i) {
          const_lstm_fwd.add_input(children[i]);
          const_lstm_rev.add_input(children[nchildren - i - 1]);
        }
        Expression cfwd = const_lstm_fwd.back();
        Expression crev = const_lstm_rev.back();
        if (apply_dropout) {
          cfwd = dropout(cfwd, DROPOUT);
          crev = dropout(crev, DROPOUT);
        }
        Expression c = concatenate({cfwd, crev});
        Expression composed = rectify(affine_transform({cbias, cW, c}));
        stack_lstm.add_input(composed);
        stack.push_back(composed);
        stack_content.push_back(curr_word);
        //cerr << curr_word << endl;
        is_open_paren.push_back(-1); // we just closed a paren at this position
      }
    }
    if (action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }
    assert(stack.size() == 2); // guard symbol, root
    Expression tot_neglogprob;
    if (!sample) {
      tot_neglogprob = -sum(log_probs);
      assert(tot_neglogprob.pg != nullptr);
      loss = tot_neglogprob;
    }
    if (sample) cerr << "\n";
    // return tot_neglogprob;
    return results;
  }

/**A structure used by beam search and particle filtering to keep track of the state of the particle/beam element*/
struct ParserState {
  LSTMBuilder stack_lstm;
  LSTMBuilder term_lstm;
  LSTMBuilder action_lstm;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;
  vector<Expression> terms;
  vector<int> is_open_paren;
  unsigned nt_count;
  vector<Expression> stack;
  vector<string> stack_content;
  vector<unsigned> results;  // sequence of predicted actions
  double score;
  int nopen_parens;
  char prev_a;
  unsigned fringe_index;
  double log_posterior_parse;
  double log_prob_additional_parse;
};

/**Provides a way to compare multiple parser states for pruning and sorting*/
struct ParserStateCompare {
  bool operator()(const ParserState* a, const ParserState* b) const {
    return a->score > b->score;
  }
};

/**Prunes a beam of ParserStates to the top k most probable elements*/
static void prune(vector<ParserState*>& pq, unsigned k) {
  if (pq.size() == 1) return;
  if (k > pq.size()) k = pq.size();
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateCompare());
  pq.resize(k);
  reverse(pq.begin(), pq.end());
}

/**Defines an action that can be applied to a parser state, and how to apply it*/
struct ParserStateAction {
  ParserState* ps;
  unsigned action;
  char a_char='0';
  unsigned wordid;
  double score;
  unsigned fringe_idx;

  /**Constructor of this structure*/
  ParserStateAction( ParserState* pstate,  unsigned a,  char achar, unsigned wid, double sc,  unsigned idx) {
    ps = pstate;
    action = a;
    a_char = achar;
    wordid = wid;
    score = sc;
    fringe_idx = idx;
  }

  /**Apply the action and return a new ParserState representing the result*/
  ParserState* applyAction(ComputationGraph* hg,  const Expression& cbias, const Expression& cW, const LookupParameter& p_a, const LookupParameter& p_w, const LookupParameter& p_nt, const LookupParameter& p_ntup, bool apply_dropout){ // p_a, p_nt, p_ntup
    ParserState* p_state = new ParserState();
    *p_state = *ps;
    p_state->fringe_index = fringe_idx;
    p_state->prev_a = a_char;
    p_state->score = score;
    p_state->log_prob_additional_parse = score ; // adiste
    p_state->results.push_back(action);

    // add current action to action LSTM
    Expression actione = lookup(*hg, p_a, action);
    p_state->action_lstm.add_input(actione);

    // do action: case NT
    if (a_char == 'N'){
      ++p_state->nopen_parens;
      auto it = action2NTindex.find(action);
      assert(it != action2NTindex.end());
      int nt_index = it->second;
      p_state->nt_count++;
      p_state->stack_content.push_back(ntermdict.convert(nt_index));
      Expression nt_embedding = lookup(*hg, p_nt, nt_index);
      p_state->stack.push_back(nt_embedding);
      p_state->stack_lstm.add_input(nt_embedding);
      p_state->is_open_paren.push_back(nt_index);
    } // do action: case REDUCE
    else if (a_char == 'R'){
      --p_state->nopen_parens;
      assert(p_state->stack.size() > 2); // dummy symbol means > 2 (not >= 2)
      assert(p_state->stack_content.size() > 2 && p_state->stack.size() == p_state->stack_content.size());
      // find what paren we are closing
      cerr << p_state->is_open_paren.size() << endl;
      int i = p_state->is_open_paren.size() - 1; //get the last thing on the stack
      while(p_state->is_open_paren[i] < 0) { --i; assert(i >= 0); } //iteratively decide whether or not it's a non-terminal
      Expression nonterminal = lookup(*hg, p_ntup, p_state->is_open_paren[i]);
      int nchildren = p_state->is_open_paren.size() - i - 1;
      assert(nchildren > 0);
      //cerr << "  number of children to reduce: " << nchildren << endl;
      vector<Expression> children(nchildren);
      p_state->const_lstm_fwd.start_new_sequence();
      p_state->const_lstm_rev.start_new_sequence();

      // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
      // TO BE COMPOSED INTO A TREE EMBEDDING
      string curr_word;
      // cerr << "--------------------------------" << endl;
      // cerr << "Now printing the children" << endl;
      // cerr << "--------------------------------" << endl;
      for (i = 0; i < nchildren; ++i) {
        assert (p_state->stack_content.size() == p_state->stack.size());
        children[i] = p_state->stack.back();
        p_state->stack.pop_back();
        p_state->stack_lstm.rewind_one_step();
        p_state->is_open_paren.pop_back();
        curr_word = p_state->stack_content.back();
        // cerr << "At the back of the stack (supposed to be one of the children): " << curr_word << endl;
        p_state->stack_content.pop_back();
      }
      assert (p_state->stack_content.size() == p_state->stack.size());
      p_state->is_open_paren.pop_back(); // nt symbol
      p_state->stack.pop_back(); // nonterminal dummy
      p_state->stack_lstm.rewind_one_step(); // nt symbol
      curr_word = p_state->stack_content.back();
      //cerr << "--------------------------------" << endl;
      //cerr << "At the back of the stack (supposed to be the non-terminal symbol) : " << curr_word << endl;
      p_state->stack_content.pop_back();
      assert (p_state->stack.size() == p_state->stack_content.size());
      // cerr << "Done reducing" << endl;
      // BUILD TREE EMBEDDING USING BIDIR LSTM
      p_state->const_lstm_fwd.add_input(nonterminal);
      p_state->const_lstm_rev.add_input(nonterminal);
      for (i = 0; i < nchildren; ++i) {
        p_state->const_lstm_fwd.add_input(children[i]);
        p_state->const_lstm_rev.add_input(children[nchildren - i - 1]);
      }
      Expression cfwd = p_state->const_lstm_fwd.back();
      Expression crev = p_state->const_lstm_rev.back();
      if (apply_dropout) {
        cfwd = dropout(cfwd, DROPOUT);
        crev = dropout(crev, DROPOUT);
      }
      Expression c = concatenate({cfwd, crev});
      Expression composed = rectify(affine_transform({cbias, cW, c}));
      p_state->stack_lstm.add_input(composed);
      p_state->stack.push_back(composed);
      p_state->stack_content.push_back(curr_word);
      p_state->is_open_paren.push_back(-1); // we just closed a paren at this position            
    }else { // apply action: SHIFT
      assert (wordid != 0);
      p_state->stack_content.push_back(termdict.convert(wordid)); //add the string of the word to the stack
      Expression word = lookup(*hg, p_w, wordid);
      p_state->terms.push_back(word);
      p_state->term_lstm.add_input(word);
      p_state->stack.push_back(word);
      p_state->stack_lstm.add_input(word);
      p_state->is_open_paren.push_back(-1);        
    }
    return p_state;
  }
};

/**Defines a way to compare two ParserStateActions*/
struct ParserStateActionCompare {
  bool operator()(const ParserStateAction* a, const ParserStateAction* b) const {
    return a->score > b->score;
  }
};

/**Prunes a beam of ParserStateActions to the top k*/
static void prune(vector<ParserStateAction*>& pq, unsigned k) {
  if (pq.size() == 1) return;
  if (k > pq.size()) k = pq.size();
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateActionCompare());
  pq.resize(k);
  reverse(pq.begin(), pq.end());
}


/**Checks to see if a proposed action is valid in generative models using beam search or particle filtering*/
bool canDoAction(const string& a, ParserState* p){
    bool is_shift = (a[0] == 'S' && a[1] == 'H');
    bool is_reduce = (a[0] == 'R' && a[1]=='E');
    bool is_nt = (a[0] == 'N');
    assert(is_shift || is_reduce || is_nt);
    if (is_reduce){
        int i = p->is_open_paren.size() -1;
        while(p->is_open_paren[i] < 0){--i;}
        if (i < 0) return false;
    }
    static const int MAX_OPEN_NTS = 100;
    if (is_nt && p->nopen_parens >= MAX_OPEN_NTS) return false;
    if (p->stack.size() == 1){
        if (!is_nt) return false;
        return true;
    }
    if (is_reduce && p->prev_a == 'N') return false;
    return true;
}

/**Probabilistic parsing using word-synchronous beam-search*/
vector<double> log_prob_parser_beam(ComputationGraph* hg,
                                    const parser::Sentence& sent,
                                    double *right,
                                    unsigned beam_size,
                                    unsigned fast_track_size,
                                    unsigned word_beam_size,
                                    bool is_evaluation) {
    //a vector to store the results in
    vector<unsigned> results;
    //store the stack in a vector, and put a dummy symbol on to begin
    vector<string> stack_content;
    stack_content.push_back("ROOT_GUARD");
    const bool sample = sent.size() == 0;
    const bool build_training_graph = true;
    assert(sample || build_training_graph);
    //check if we're applying dropout and apply it if we are
    bool apply_dropout = (DROPOUT && !is_evaluation && !sample);
    if (apply_dropout) {
      stack_lstm.set_dropout(DROPOUT);
      term_lstm.set_dropout(DROPOUT);
      action_lstm.set_dropout(DROPOUT);
      const_lstm_fwd.set_dropout(DROPOUT);
      const_lstm_rev.set_dropout(DROPOUT);
    } else {
      stack_lstm.disable_dropout();
      term_lstm.disable_dropout();
      action_lstm.disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
    }

    //initialize the computation graphs here
    term_lstm.new_graph(*hg);
    stack_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    cfsm->new_graph(*hg);
    term_lstm.start_new_sequence();
    stack_lstm.start_new_sequence();
    action_lstm.start_new_sequence();

    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression A = parameter(*hg, p_A);
    Expression T = parameter(*hg, p_T);
    Expression cbias = parameter(*hg, p_cbias);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);

    //initialize the action and term lstms (?)
    action_lstm.add_input(action_start);
    vector<Expression> terms(1, lookup(*hg, p_w, kSOS));
    term_lstm.add_input(terms.back());

    vector<Expression> stack;  // variables representing subtree embeddings
    stack.push_back(parameter(*hg, p_stack_guard));
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    int nopen_parens = 0; // there are no open parentheses, and the first action is a dummy
    char prev_a = '0';

    //create a new parser state object and assign the relevant values
    ParserState* init = new ParserState();
    init->stack_lstm = stack_lstm;
    init->term_lstm = term_lstm;
    init->action_lstm = action_lstm;
    init->const_lstm_fwd = const_lstm_fwd;
    init->const_lstm_rev = const_lstm_rev;
    init->terms = terms;
    init->is_open_paren = is_open_paren;
    init->nt_count = 0;
    init->stack = stack;
    init->stack_content = stack_content;
    init->results = results;
    // init.log_probs = log_probs;
    init->score = 0;
    init->nopen_parens = nopen_parens;
    init->prev_a = prev_a;
    init->fringe_index = 0;
    init->log_posterior_parse = 0;
    init->log_prob_additional_parse = 0;

    //keep track of the parser states and the ones we need to delete
    vector<ParserState*> pq_this;
    pq_this.push_back(init);
    unordered_set<ParserState*> need_to_delete;

    //store the surprisals and log probabilities
    vector<double> surprisals;
    vector<double> log_probs;

    //iterate through the sentence
    for (unsigned w_index = 0; w_index < sent.size(); ++w_index) {
      cerr << "Word index: " << w_index << " " << termdict.convert(sent.raw[w_index]) << "; thiswords size: " << pq_this.size() << endl;
      vector<ParserState*> pq_next;

      //while the beam isn't full, expand it
      while (pq_next.size() < beam_size){
        vector<ParserStateAction*> fringe;
        // iterate through all parser states in the input beam
        for (auto p_this : pq_this){
          assert (p_this->stack.size() == p_this->stack_content.size());
          //get all current valid actions at the parser state
          vector<unsigned> current_valid_actions;
          for (auto a: possible_actions) {
              if (canDoAction(adict.convert(a), p_this)) current_valid_actions.push_back(a);
          }
          //get the corresponding stack, action, and term summaries, and apply dropout if needed
          Expression stack_summary = p_this->stack_lstm.back();
          Expression action_summary = p_this->action_lstm.back();
          Expression term_summary = p_this->term_lstm.back();
          if (apply_dropout) {
            stack_summary = dropout(stack_summary, DROPOUT);
            action_summary = dropout(action_summary, DROPOUT);
            term_summary = dropout(term_summary, DROPOUT);
          }

          Expression p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
          Expression nlp_t = rectify(p_t);
          Expression r_t = affine_transform({abias, p2a, nlp_t});
          //action distribution expression over the current valid actions
          Expression adiste = log_softmax(r_t, current_valid_actions);

          //iterate through the current valid actions and add them to the fringe
          for(auto action : current_valid_actions){
	        const string& a_string = adict.convert(action);
	        char a_char = a_string[0];
	        double new_score = p_this->score + as_scalar(pick(adiste, action).value());
	        unsigned wordid = 0;
	        // class factored softmax if we're doing generation
	        wordid = sent.raw[w_index];
	        if (a_char == 'S'){
	            new_score += as_scalar((-cfsm->neg_log_softmax(nlp_t, wordid)).value());}
	        ParserStateAction* p_state_action = new ParserStateAction(p_this, action, a_char, wordid, new_score,  fringe.size());
	        fringe.push_back(p_state_action);
          }
	    }
        // sort all parser states in the fringe
        prune(fringe, fringe.size());
        unsigned fast_track_count = 0;
        int cut = max<int>((int)fringe.size()-beam_size, 0);


	    vector<ParserState*> pq_this_new;
	    //iterate through the fringe
        for(int k = fringe.size()-1; k >= 0; k--){
            //if the item is above the cut
            if(k >= cut){
	            ParserState* newstate = fringe[k]->applyAction(hg, cbias, cW, p_a, p_w, p_nt, p_ntup, apply_dropout);
                //if came by lexical action, then add it to possible next actions
	            if(fringe[k]->a_char == 'S'){
                    pq_next.push_back(newstate);
                //otherwise, structural, and add to the new pq_next
                }else{
                    pq_this_new.push_back(newstate);
                }
	        //if it's not above the cut and it's a lexical action, only add if we're fast-tracking
            } else {
                if(fringe[k]->a_char == 'S' && fast_track_count < fast_track_size){
	                ParserState* newstate = fringe[k]->applyAction(hg, cbias, cW, p_a, p_w, p_nt, p_ntup, apply_dropout);
                    pq_next.push_back(newstate);
                    fast_track_count += 1;
                }
            }
        }

	    //clear the fringe and update the current action queue
        for (ParserStateAction* psa: fringe) {delete psa;}
        fringe.clear();
        fringe.shrink_to_fit();
        for (ParserState* ps: pq_this) {delete ps;}
        pq_this.clear();
        pq_this.shrink_to_fit();
	    for (auto p_new : pq_this_new) {
	        pq_this.push_back(p_new);
	    }
	    pq_this_new.clear();
	    pq_this_new.shrink_to_fit();

        int delcount = 0;
        for (ParserState* ps: need_to_delete) {delete ps;delcount++;}
        need_to_delete.clear();
      }

      // add parser state pointers from pq_next to need_to_delete
      for (auto parser_state : pq_next){
        need_to_delete.insert(parser_state);
      }

      // sort and prune parser states according to their forward probs from low to high, and then print a list of parses
      prune(pq_next, word_beam_size);
      cerr << "list of partial parses:" << endl;
      unsigned beam_index = 1;
      for(auto parser_state : pq_next){
        cerr << beam_index << " " << parser_state->score << " ";
        beam_index++;
        unsigned shift_count = 0;
        for (auto action : parser_state->results){
          const string& a = adict.convert(action);
          if (a[0] == 'R') cerr << ")";
          if (a[0] == 'N') {
            int nt = action2NTindex[action];
            cerr << " (" << ntermdict.convert(nt);
          }     
          if (a[0] == 'S'){
            cerr << " " << termdict.convert(sent.raw[shift_count]);
            shift_count++; 
          }
        }
        cerr << " " << parser_state->stack.size() << "\n";
      }      

      //clear the current states, only need the next ones
      for (ParserState* ps: pq_this) {delete ps;}
      pq_this.clear();
      pq_this.shrink_to_fit();

      double marginal_prob = 0;
      for (auto ps : pq_next){
        marginal_prob += exp(ps->score);
      }
      //get the log probabilities, and update pq_this
      log_probs.push_back(-log2(marginal_prob));
      for (auto ps : pq_next){
        ParserState* p_new = new ParserState();
        *p_new = *ps;
        pq_this.push_back(p_new);
      }
      int delcount = 0;
      for (ParserState* ps: need_to_delete) {delete ps; delcount++;}
      need_to_delete.clear();
    }

    //calculate surprisals
    for (unsigned k = 0; k < log_probs.size(); k++){
      if(k == 0){
        surprisals.push_back(log_probs[k]);
      }else{
        surprisals.push_back(log_probs[k] - log_probs[k-1]);
      }
    }
    // print words and surprisals
    for (unsigned k = 0; k < surprisals.size(); k++){
      cerr << termdict.convert(sent.raw[k])  << "\t" << surprisals[k] << endl;
    }
    //clear the queue and return the surprisals
    for (ParserState* ps: pq_this) {delete ps;}
    pq_this.clear();
    return surprisals;
    }


vector<double> log_prob_parser_particle(ComputationGraph* hg,
                                        const parser::Sentence& sent,
                                        double *right,
                                        unsigned num_particles,
                                        bool is_evaluation) {
        //make sure that the sentence isn't empty
        assert(sent.size() > 0);
        //check if we're applying dropout and apply it if we are
        bool apply_dropout = (DROPOUT && !is_evaluation);
        if (apply_dropout) {
            stack_lstm.set_dropout(DROPOUT);
            term_lstm.set_dropout(DROPOUT);
            action_lstm.set_dropout(DROPOUT);
            const_lstm_fwd.set_dropout(DROPOUT);
            const_lstm_rev.set_dropout(DROPOUT);
        } else {
            stack_lstm.disable_dropout();
            term_lstm.disable_dropout();
            action_lstm.disable_dropout();
            const_lstm_fwd.disable_dropout();
            const_lstm_rev.disable_dropout();
        }

        //initialize the computation graphs here
        term_lstm.new_graph(*hg);
        stack_lstm.new_graph(*hg);
        action_lstm.new_graph(*hg);
        const_lstm_fwd.new_graph(*hg);
        const_lstm_rev.new_graph(*hg);
        cfsm->new_graph(*hg);
        term_lstm.start_new_sequence();
        stack_lstm.start_new_sequence();
        action_lstm.start_new_sequence();

        // variables in the computation graph representing the parameters
        Expression pbias = parameter(*hg, p_pbias);
        Expression S = parameter(*hg, p_S);
        Expression A = parameter(*hg, p_A);
        Expression T = parameter(*hg, p_T);
        Expression cbias = parameter(*hg, p_cbias);
        Expression p2a = parameter(*hg, p_p2a);
        Expression abias = parameter(*hg, p_abias);
        Expression action_start = parameter(*hg, p_action_start);
        Expression cW = parameter(*hg, p_cW);
        //add the inputs to the LSTMs
        action_lstm.add_input(action_start);
        term_lstm.add_input(lookup(*hg, p_w, kSOS));
        stack_lstm.add_input(parameter(*hg, p_stack_guard));

        //a vector in which we'll store the particles
        vector<ParserState*> particles;
        //intialize each particle
        for (unsigned i = 0; i < num_particles; i++){
            vector<unsigned> results;
            // -1 if no nonterminal has a parenthesis open, otherwise index of NT
            vector<int> is_open_paren;
            is_open_paren.push_back(-1);
            // variables representing subtree embeddings
            vector<Expression> stack;
            //push the stack guard onto the stack
            stack.push_back(parameter(*hg, p_stack_guard));
            //initialize the stack and term lstms
            vector<Expression> terms(1, lookup(*hg, p_w, kSOS));
            vector<string> stack_content;
            stack_content.push_back("ROOT_GUARD");
            ParserState* currstate = new ParserState();
            currstate->stack_lstm = stack_lstm;
            currstate->term_lstm = term_lstm;
            currstate->action_lstm = action_lstm;
            currstate->const_lstm_fwd = const_lstm_fwd;
            currstate->const_lstm_rev = const_lstm_rev;
            currstate->terms = terms;
            currstate->is_open_paren = is_open_paren;
            currstate->nt_count = 0;
            currstate->stack = stack;
            currstate->stack_content = stack_content;
            currstate->results = results;
            currstate->score = 0;
            currstate->nopen_parens = 0;
            currstate->prev_a = '0';
            currstate->fringe_index = 0;
            currstate->log_posterior_parse = 0;
            currstate->log_prob_additional_parse = 0;
            particles.push_back(currstate);
        }

        //store the surprisals to return at the end
        vector<double> surprisals;

        //iterate through the sentence
        for (unsigned w_index = 0; w_index < sent.size(); ++w_index) {
            cerr << "Current word: " << termdict.convert(sent.raw[w_index]) << " ,index " << w_index << endl;
            cerr << "List of partial parses: \n";
            for (unsigned y = 0; y < num_particles; y++) {
                //repeatedly sample until we reach a shift operation
                char a_char = '0';
                while (a_char != 'S'){
                    //get the valid actions for the particle
                    vector<unsigned> current_valid_actions;
                    for (auto a: possible_actions) {
                        if (canDoAction(adict.convert(a), particles[y])) current_valid_actions.push_back(a);
                    }
                    //get the stack, action, and term summaries from the LSTMs, applying dropout if needed
                    Expression stack_summary = particles[y]->stack_lstm.back();
                    Expression action_summary = particles[y]->action_lstm.back();
                    Expression term_summary = particles[y]->term_lstm.back();
                    if (apply_dropout) {
                        stack_summary = dropout(stack_summary, DROPOUT);
                        action_summary = dropout(action_summary, DROPOUT);
                        term_summary = dropout(term_summary, DROPOUT);
                    }

                    //get the action distribution expression over the current valid actions
                    Expression p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
                    Expression nlp_t = rectify(p_t);
                    Expression r_t = affine_transform({abias, p2a, nlp_t});
                    Expression adiste = log_softmax(r_t, current_valid_actions);

                    //select an action to pursue by sampling from the distribution of actions using random number generation
                    float random = rand01();
                    float total = 0;
                    int action = -1;
                    for (unsigned i = 0; i < current_valid_actions.size(); i++){
                        total += exp(as_scalar(pick(adiste, current_valid_actions[i]).value()));
                        if (random <= total){ action = current_valid_actions[i]; break;}
                    }
                    // set the character to correspond to the action
                    a_char = adict.convert(action)[0];
                    double new_score = 0;
                    unsigned wordid = sent.raw[w_index];
                    //re-weight by P(w_t|y) if the operation is a shift
                    if (a_char == 'S'){new_score = as_scalar((-cfsm->neg_log_softmax(nlp_t, wordid)).value());}
                    //apply the action and update the particle so we can proceed until we reach a shift
                    ParserStateAction* p_state_action = new ParserStateAction(particles[y], action, a_char, wordid, new_score, 0);
                    ParserState* newstate = p_state_action->applyAction(hg, cbias, cW, p_a, p_w, p_nt, p_ntup, apply_dropout);
                    particles[y] = newstate;
                    assert(particles[y]->log_prob_additional_parse == new_score);
                }
                // print out the partial parses
                int shift_count = 0;
                for (auto action : particles[y]->results){
                    const string& a = adict.convert(action);
                    if (a[0] == 'R') cerr << ")";
                    if (a[0] == 'N') {
                        int nt = action2NTindex[action];
                        cerr << " (" << ntermdict.convert(nt);
                    }
                    if (a[0] == 'S'){
                        cerr << " " << termdict.convert(sent.raw[shift_count]);
                        shift_count++;
                    }
                }
                cerr << exp(particles[y]->log_prob_additional_parse) << " " << particles[y]->stack.size();
                cerr << endl;
            }
            //resample the particles based on the updated weights
            vector<ParserState*> resampled;
            //get the total for re-normalization and for computing surprisal
            float total = 0;
            for (unsigned i = 0; i < num_particles; i++){total += exp(particles[i]->log_prob_additional_parse);}
            //sample for each particle by generating a random number between 0 and 1
            for (unsigned p = 0; p < num_particles; p++){
                float partial_total = 0;
                float random = rand01();
                for (unsigned i = 0; i < num_particles; i++){
                    partial_total += exp(particles[i]->log_prob_additional_parse)/total;
                    if (random <= partial_total){ resampled.push_back(particles[i]); break;}
                }
            }
            //surprisal = log(N/sum(P(w|C)))
            surprisals.push_back(log(num_particles/total));
            assert(particles.size() == resampled.size());
            //update the particles based on the values we have just resampled
            particles.clear();
            for (ParserState* re : resampled){particles.push_back(re);}
            resampled.clear();
        }
        return surprisals;
    }
};

/**Signal handler for graceful termination*/
void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);
    //initialize the command-line arguments into a map from the variables to their values
    po::variables_map conf;
    InitCommandLine(argc, argv, &conf);
    //clusters are need for computational efficiency
    if (!conf.count("clusters")) {
        cerr << "Please specify vocabulary clustering with --clusters FILE when training generative model\n";
        return 1;
    }
    //need dev data if we're training the model
    if (conf.count("train") && conf.count("dev_data") == 0) {
        cerr << "You specified --train but did not specify --dev_data FILE\n";
        return 1;
    }
    if (conf.count("dropout")) DROPOUT = conf["dropout"].as<float>();
    LAYERS = conf["layers"].as<unsigned>();
    INPUT_DIM = conf["input_dim"].as<unsigned>();
    PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
    HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
    ACTION_DIM = conf["action_dim"].as<unsigned>();
    LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
    BEAM_SIZE = conf["beam_size"].as<unsigned>();
    FASTTRACK_BEAM_SIZE = conf["fasttrack_beam_size"].as<unsigned>();
    WORD_BEAM_SIZE = conf["word_beam_size"].as<unsigned>();
    LEARNING_RATE = conf["lr"].as<float>();
    NUM_PARTICLES = conf["np"].as<unsigned>();

    //print out the name of the model that will result
    ostringstream os;
    os << "ntparse_gen"
     << "_D" << DROPOUT
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "PARAMETER FILE: " << fname << endl;
    bool softlinkCreated = false;

    //initialize the model and the cfsm for fast lookup
    kSOS = termdict.convert("<s>");
    ParameterCollection model;
    cfsm = new ClassFactoredSoftmaxBuilder(HIDDEN_DIM, conf["clusters"].as<string>(), termdict, model);

    //load in the corpora as oracles
    parser::TopDownOracleGen corpus(&termdict, &adict, &posdict, &ntermdict);
    parser::TopDownOracleGen dev_corpus(&termdict, &adict, &posdict, &ntermdict);
    parser::TopDownOracleGen2 test_corpus(&termdict, &adict, &posdict, &ntermdict);
    parser::TopDownOracleGen eval_corpus(&termdict, &adict, &posdict, &ntermdict);
    corpus.load_oracle(conf["training_data"].as<string>());
    if (conf.count("words"))
        parser::ReadEmbeddings_word2vec(conf["words"].as<string>(), &termdict, &pretrained);

    // freeze dictionaries so we don't accidentaly load OOVs
    termdict.freeze();
    adict.freeze();
    ntermdict.freeze();
    posdict.freeze();

    //read in all the data
    if (conf.count("dev_data")) {
        cerr << "Loading validation set\n";
        dev_corpus.load_oracle(conf["dev_data"].as<string>());
    }
    if (conf.count("test_data")) {
        cerr << "Loading test set\n";
        test_corpus.load_oracle(conf["test_data"].as<string>());
    }
    if (conf.count("eval_data")) {
        cerr << "Loading evaluation set\n";
        eval_corpus.load_raw_sent(conf["eval_data"].as<string>());
    }

    // linking the action dictionary to the non-terminal dictionary: pass in index of action NT(X), return index of X
    for (unsigned i = 0; i < adict.size(); ++i) {
        const string& a = adict.convert(i);
        if (a[0] != 'N') continue;
        size_t start = a.find('(') + 1;
        size_t end = a.rfind(')');
        int nt = ntermdict.convert(a.substr(start, end - start));
        action2NTindex[i] = nt;
    }

    //the dictionary of non-terminals that can be introduced onto the stack
    NT_SIZE = ntermdict.size();
    //the dictionary of vocabulary in the training & dev sets
    VOCAB_SIZE = termdict.size();
    //the dictionary of actions that the parser can take
    ACTION_SIZE = adict.size();
    //right now, the possible actions are all actions
    possible_actions.resize(adict.size());
    for (unsigned i = 0; i < adict.size(); ++i)
      possible_actions[i] = i;

    //load in existing model
    ParserBuilder parser(&model, pretrained);
    if (conf.count("model")) {
        cerr << "Loading existing model: " << conf["model"].as<string>().c_str() << "\n";
        dynet::TextFileLoader l(conf["model"].as<string>().c_str());
        l.populate(model);
    }

  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    //AdamTrainer sgd(&model);
    SimpleSGDTrainer sgd(model, LEARNING_RATE);
    //sgd.eta = 0.01;
    //sgd.eta0 = 0.01;
    //MomentumSGDTrainer sgd(&model);
    //how often to log the status of training
    unsigned status_every_i_iterations = 100;
    double eta_decay = 0.08;
    //sgd.eta_decay = 0.05;
    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min((int)status_every_i_iterations, (int)corpus.sents.size());
    unsigned si = corpus.sents.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.sents.size() << endl;
    unsigned trs = 0;
    unsigned words = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    double best_dev_llh = 9e99;
    //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
    while(!requested_stop) {
      ++iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.sents.size()) {
             si = 0;
             if (first) { first = false; } else {
                  sgd.update_epoch();
                  //sgd.eta /= 2;
                  sgd.learning_rate /= (1 + eta_decay);
              }
             //cerr << "NO SHUFFLE" << endl;
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
             //sgd.eta /= 2;
           }
           tot_seen += 1;
           auto& sentence = corpus.sents[order[si]];
	   const vector<int>& actions=corpus.actions[order[si]];
           ComputationGraph hg;
           Expression tot_neglogprob;
           parser.log_prob_parser(&hg,tot_neglogprob,sentence,actions,&right,false);
           
           double lp = as_scalar(hg.incremental_forward(tot_neglogprob));
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward(tot_neglogprob);
           // sgd.update(1.0);
           sgd.update();
           llh += lp;
           ++si;
           trs += actions.size();
           words += sentence.size();

          // if (iter > 15){
          //   cerr << "Testing beam search..." << endl;
          //   parser.log_prob_parser_beam(&hg,sentence,&right, 50, false);
          //   cerr << "Complete one sentence..." << endl;
          // }

      }
      sgd.status();
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<
         /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
        ") per-action-ppl: " << exp(llh / trs) << " per-input-ppl: " << exp(llh / words) << " per-sent-ppl: " << exp(llh / status_every_i_iterations) << " err: " << (trs - right) / trs << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;
      llh = trs = right = words = 0;
      static int logc = 0;
      ++logc;
      if (logc > 50) {
        // generate random sample
        ComputationGraph cg;
        double x;
        Expression tot_neglogprob;
        parser.log_prob_parser(&cg, tot_neglogprob, parser::Sentence(), vector<int>(),&x,true);
      }
      if (logc % 100 == 0) { // report on dev set
        unsigned dev_size = dev_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const auto& sentence=dev_corpus.sents[sii];
	   const vector<int>& actions=dev_corpus.actions[sii];
           dwords += sentence.size();
           {  ComputationGraph hg;
              Expression tot_neglogprob;
              parser.log_prob_parser(&hg,tot_neglogprob,sentence,actions,&right,true);
              double lp = as_scalar(hg.incremental_forward(tot_neglogprob));
              llh += lp;
           }
           trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        double err = (trs - right) / trs;
        //parser::EvalBResults res = parser::Evaluate("foo", pfx);
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
        if (llh < best_dev_llh && (tot_seen / corpus.size()) > 1.0) {
          cerr << "  new best...writing model to " << fname << " ...\n";
          best_dev_llh = llh;

          dynet::TextFileSaver out(fname);
          out.save(model);

          // ofstream out(fname);
          // boost::archive::text_oarchive oa(out);
          // oa << model;

          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
        }
      }
    }
  } // should do training?

  //TESTING
  if (test_corpus.size() > 0) {
    // if rescoring, we may have many repeats, cache them
    unordered_map<vector<int>, unordered_map<vector<int>,double,boost::hash<vector<int>>>, boost::hash<vector<int>>> s2a2p;
    unsigned test_size = test_corpus.size();
    double llh = 0;
    double right = 0;
    double dwords = 0;
    for (unsigned sii = 0; sii < test_size; ++sii) {
      const auto& sentence=test_corpus.sents[sii];
      const vector<int>& actions=test_corpus.actions[sii];
      dwords += sentence.size();
      double& lp = s2a2p[sentence.raw][actions];
      if (!lp) {
        ComputationGraph hg;
        Expression tot_neglogprob;
        parser.log_prob_parser(&hg,tot_neglogprob,sentence,actions,&right,true);
        lp = as_scalar(hg.incremental_forward(tot_neglogprob));
      }
      cout << sentence.size() << '\t' << lp << endl;
      llh += lp;
    }
    cerr << "test     total -llh=" << llh << endl;
    cerr << "test ppl (per word)=" << exp(llh / dwords) << endl;
  }

  //EVALUATION & SURPRISAL
  if (eval_corpus.size() > 0) {
      cerr << "Beginning evaluation...\n";
      auto t_start = std::chrono::high_resolution_clock::now();
      unordered_map<vector<int>, unordered_map<vector<int>,double,boost::hash<vector<int>>>, boost::hash<vector<int>>> s2a2p;
      unsigned eval_size = eval_corpus.size();
      double right = 0;

      //create the file stream to write to
      ofstream f;
      string f_name;
      if (conf.count("file_path")){
        f_name = conf["file_path"].as<string>();
      }else{
        f_name = "./surprisals_"+conf["eval_data"].as<string>()+".txt";
      }
      f.open(f_name);
      // output header
      f << "sentence_id\ttoken_id\ttoken\tsurprisal\n";

      //iterate through the sentences
      for (unsigned sii = 0; sii < eval_size; ++sii) {
        const auto& sentence=eval_corpus.sents[sii];
        ComputationGraph hg;
        vector<double> surprisals;
        bool run_beam = conf["beam"].as<bool>();
        if (run_beam){
            cerr << "Running beam search with beam size " << BEAM_SIZE << endl;
            surprisals  = parser.log_prob_parser_beam(&hg, sentence, &right, BEAM_SIZE, FASTTRACK_BEAM_SIZE, WORD_BEAM_SIZE, false);
        }
        else {
            cerr << "Running particle filtering with " << NUM_PARTICLES << " particles\n";
            surprisals = parser.log_prob_parser_particle(&hg, sentence, &right, NUM_PARTICLES, false);
        }
        //write out the surprisals
        for(unsigned k = 0; k < surprisals.size(); ++k){
            f << (sii + 1) << "\t" << (k + 1) << "\t" << termdict.convert(sentence.raw[k]) << "\t" << surprisals[k] <<"\n";
        }
        f.flush();
      }
      //close the file, and log the time taken to complete
      f.close();
      cerr << "Results written to: " << f_name << "...\n";
      auto t_end = std::chrono::high_resolution_clock::now();
      cerr << "TEST: " << eval_corpus.size()  << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
  }
}
