from oxlm.config import ModelData
from oxlm.corpus import Dict

import theano
import theano.tensor as T
import numpy as np

class CNLMBase(object):
    """docstring for CNLMBase"""
    def __init__(self, config=None, target_labels=None, classes=None):
        super(CNLMBase, self).__init__()
        if config is not None:
            if type(config) is not ModelData:
                pass
            if type(target_labels) is not Dict:
                pass
            if type(classes) is not list:
                pass

            # General INIT is: R(0,0,0), Q(0,0,0), F(0,0,0), B(0,0), FB(0,0), W(0,0), m_data(0)
            # Public members NEED INIT
            self.config = config # ModelData config;

            self.ContextTransforms = None # ContextTransformsType C;  // Context position transforms
            self.outputWordVectors = None # WordVectorsType       R;  // output word representations
            self.contextWordVectors = None # WordVectorsType       Q;  // context word representations
            self.classVectors = None # WordVectorsType       F;  // class representations
            self.outputWordBiases = None # WeightsType           B;  // output word biases
            self.outputClassBiases = None # WeightsType           FB; // output class biases

            self.weights = None # WeightsType           W;  // All the parameters in one vector
            self.length_ratio = 1 # Real length_ratio;

            # Private members NEED INIT
            self._m_target_labels = target_labels # Dict m_target_labels;
            self._m_data_size = None # int m_data_size;
            self._m_data = None # Real* m_data;

            self._word_to_class = None # std::vector<int> word_to_class; // map from word id to class
            self._indexes = classes # std::vector<int> indexes;       // vocab spans for each class

        else:
            # General INIT is: R(0,0,0), Q(0,0,0), F(0,0,0), B(0,0), FB(0,0), W(0,0), m_data(0)
            # Public members NEED INIT
            self.config = None # ModelData config;

            self.ContextTransforms = None # ContextTransformsType C;  // Context position transforms
            self.outputWordVectors = None # WordVectorsType       R;  // output word representations
            self.contextWordVectors = None # WordVectorsType       Q;  // context word representations
            self.classVectors = None # WordVectorsType       F;  // class representations
            self.outputWordBiases = None # WeightsType           B;  // output word biases
            self.outputClassBiases = None # WeightsType           FB; // output class biases

            self.weights = None # WeightsType           W;  // All the parameters in one vector
            self.length_ratio = None # Real length_ratio;

            # Private members NEED INIT
            self._m_target_labels = None # Dict m_target_labels;
            self._m_data_size = None # int m_data_size;
            self._m_data = None # Real* m_data;

            self._word_to_class = None # std::vector<int> word_to_class; // map from word id to class
            self._indexes = None # std::vector<int> indexes;       // vocab spans for each class

    def build_model(self, word_ids, num_tokens, order, word_width):
        contexts = T.imatrix(np.zeros((num_tokens, order), dtype=int))

# def function(self):
# """Originally:

# virtual void init(bool init_weights=false);
# """
# pass

# def function(self):
# """Originally:

# virtual int calculateDataSize(bool allocate=false);
# """
# pass

# def function(self):
# """Originally:

# void initWordToClass();
# """
# pass

# def function(self):
# """Originally:

# int output_types() const { return m_target_labels.size(); }
# """
# pass

# def function(self):
# """Originally:

# int context_types() const { return m_target_labels.size(); }
# """
# pass

# def function(self):
# """Originally:

# int labels() const { return m_target_labels.size(); }
# """
# pass

# def function(self):
# """Originally:

# const Dict& label_set() const { return m_target_labels; }
# """
# pass

# def function(self):
# """Originally:

# Dict& label_set() { return m_target_labels; }
# """
# pass

# def function(self):
# """Originally:

# Real l2_gradient_update(Real sigma) {
#   W -= W*sigma;
#   return W.squaredNorm();
# }
# """
# pass

# def function(self):
# """Originally:

# WordId label_id(const Word& l) const { return m_target_labels.Lookup(l); }
# """
# pass

# def function(self):
# """Originally:

# const Word& label_str(WordId i) const { return m_target_labels.Convert(i); }
# """
# pass

# def function(self):
# """Originally:

# virtual int num_weights() const { return m_data_size; }
# """
# pass

# def function(self):
# """Originally:

# Real* data() { return m_data; }
# """
# pass

# def function(self):
# """Originally:

# Real gradient_(
#     const std::vector<Sentence>& target_corpus,
#     const TrainingInstances& training_instances,
#     // std::function<void(TrainingInstance, VectorReal)> source_repr_callback,
#     // std::function<void(TrainingInstance, int, int, VectorReal)>
#     //   source_grad_callback,
#     Real l2, Real source_l2, Real*& g_ptr);
# """
# pass

# def function(self):
# """Originally:

# virtual void source_repr_callback(TrainingInstance t, int t_i,
#                                   VectorReal& r) = 0;
# """
# pass

# def function(self):
# """Originally:

# virtual void source_grad_callback(TrainingInstance t, int t_i,
#                                   int instance_counter,
#                                   const VectorReal& grads) = 0;
# """
# pass

# def function(self):
# """Originally:

# void source_representation(const Sentence& source, int target_index,
#                            VectorReal& result) const;
# """
# pass

# def function(self):
# """Originally:

# void hidden_layer(const std::vector<WordId>& context,
#                   const VectorReal& source, VectorReal& result) const;
# """
# pass

# def function(self):
# """Originally:

# Real log_prob(const WordId w, const std::vector<WordId>& context,
#               bool cache=false) const;
# """
# pass

# def function(self):
# """Originally:

# Real log_prob(const WordId w, const std::vector<WordId>& context,
#               const VectorReal& source, bool cache=false) const;
# """
# pass

# def function(self):
# """Originally:

# void class_log_probs(const std::vector<WordId>& context,
#                      const VectorReal& source,
#                      const VectorReal& prediction_vector, VectorReal& result,
#                      bool cache=false) const;
# """
# pass

# def function(self):
# """Originally:

# void word_log_probs(int c, const std::vector<WordId>& context,
#                     const VectorReal& source,
#                     const VectorReal& prediction_vector, VectorReal& result,
#                     bool cache=false) const;
# """
# pass

# def function(self):
# """Originally:

# Eigen::Block<WordVectorsType> class_R(const int c) {
#   int c_start = indexes.at(c), c_end = indexes.at(c+1);
#   return R.block(c_start, 0, c_end-c_start, R.cols());
# }
# """
# pass

# def function(self):
# """Originally:

# const Eigen::Block<const WordVectorsType> class_R(const int c) const {
#   int c_start = indexes.at(c), c_end = indexes.at(c+1);
#   return R.block(c_start, 0, c_end-c_start, R.cols());
# }
# """
# pass

# def function(self):
# """Originally:

# Eigen::VectorBlock<WeightsType> class_B(const int c) {
#   int c_start = indexes.at(c), c_end = indexes.at(c+1);
#   return B.segment(c_start, c_end-c_start);
# }
# """
# pass

# def function(self):
# """Originally:

# const Eigen::VectorBlock<const WeightsType> class_B(const int c) const {
#   int c_start = indexes.at(c), c_end = indexes.at(c+1);
#   return B.segment(c_start, c_end-c_start);
# }
# """
# pass

# def function(self):
# """Originally:

# int get_class(const WordId& w) const {
#   assert(w >= 0 && w < int(word_to_class.size())
#          && "ERROR: Failed to find word in class dictionary.");
#   return word_to_class[w];
# }
# """
# pass

# def function(self):
# """Originally:

# int map_class_to_word_index(int c, int wc) const {
#   int c_start = indexes.at(c);
#   return wc + c_start;
# }
# """
# pass

# def function(self):
# """Originally:

# void clear_cache() {
#   m_context_cache.clear();
#   m_context_cache.reserve(1000000);
#   m_context_class_cache.clear();
#   m_context_class_cache.reserve(1000000);
# }
# """

# def function(self):
# """Originally:

# void map_parameters(Real*& ptr, WordVectorsType& r, WordVectorsType& q,
#                     WordVectorsType& f, ContextTransformsType& c,
#                     WeightsType& b, WeightsType& fb) const;
# """

# """Originally:

# MatrixReal context_product(int i, const MatrixReal& v,
#                            bool transpose=false) const {
#   if (config.diagonal)
#     return (C.at(i).asDiagonal() * v.transpose()).transpose();
#   else if (transpose) return v * C.at(i).transpose();
#   else                return v * C.at(i);
# }
# """
# pass

# def function(self):
# """Originally:

# void context_gradient_update(ContextTransformType& g_C, const MatrixReal& v,
#                              const MatrixReal& w) const {
#   if (config.diagonal) g_C += (v.cwiseProduct(w).colwise().sum()).transpose();
#   else                 g_C += (v.transpose() * w);
# }
# """
# pass