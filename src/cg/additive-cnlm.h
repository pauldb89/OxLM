
  MatrixReal window_product(int i, const MatrixReal& v,
                            bool transpose=false) const {
    if (config.diagonal)
      return (T.at(i).asDiagonal() * v.transpose()).transpose();
    else if (transpose) return v * T.at(i).transpose();
    else                return v * T.at(i);
  }


  // Allocate only own parameters (not base class weights).
  void map_parameters(Real*& ptr, WordVectorsType& s,
                      ContextTransformsType& t) const;

  // Lazy function: allocate own and subsequently parent parameters.

  /*virtual*/void init(bool init_weights=false);
protected:
  /*virtual*/int calculateDataSize(bool allocate=false);

  Dict m_source_labels;


typedef std::shared_ptr<AdditiveCNLM> AdditiveCNLMPtr;

}  // namespace oxlm
#endif  // CG_ADDITIVE_CNLM_H
