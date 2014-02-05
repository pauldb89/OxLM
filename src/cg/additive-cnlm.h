  // Better name needed. Make source_corpus const again.
  Real gradient(std::vector<Sentence>& source_corpus,
                const std::vector<Sentence>& target_corpus,
                const TrainingInstances &training_instances,
                Real l2, Real source_l2, WeightsType& g_W);

  Real log_prob(const WordId w, const std::vector<WordId>& context,
                const Sentence& source, bool cache=false,
                int target_index=-1) const;

  MatrixReal window_product(int i, const MatrixReal& v,
                            bool transpose=false) const {
    if (config.diagonal)
      return (T.at(i).asDiagonal() * v.transpose()).transpose();
    else if (transpose) return v * T.at(i).transpose();
    else                return v * T.at(i);
  }

  void source_representation(const Sentence& source, int target_index,
                             VectorReal& result) const;

  // Allocate only own parameters (not base class weights).
  void map_parameters(Real*& ptr, WordVectorsType& s,
                      ContextTransformsType& t) const;

  // Lazy function: allocate own and subsequently parent parameters.
  ContextTransformsType T;  // source window context transforms
  WordVectorsType       S;  // source word representations

  ContextTransformsType g_T;  // source window context transforms
  WordVectorsType       g_S;  // source word representations

  std::vector<Sentence> source_corpus;

  /*virtual*/void init(bool init_weights=false);
protected:
  /*virtual*/int calculateDataSize(bool allocate=false);

  Dict m_source_labels;


typedef std::shared_ptr<AdditiveCNLM> AdditiveCNLMPtr;

}  // namespace oxlm
#endif  // CG_ADDITIVE_CNLM_H
