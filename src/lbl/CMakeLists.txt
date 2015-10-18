#############################################
# Libraries
#############################################

add_libraries(lbl
  class_context_hasher.cc
  class_distribution.cc
  class_tree.cc
  config.cc
  context_cache.cc
  context_processor.cc
  corpus.cc
  factored_maxent_metadata.cc
  factored_metadata.cc
  factored_tree_weights.cc
  factored_weights.cc
  feature_context.cc
  feature_context_generator.cc
  feature_context_hasher.cc
  feature_filter.cc
  feature_indexes_pair.cc
  feature_matcher.cc
  feature_store.cc
  global_collision_space.cc
  global_factored_maxent_weights.cc
  global_feature_store.cc
  hash_builder.cc
  hashed_ngram.cc
  metadata.cc
  minibatch_factored_maxent_weights.cc
  minibatch_feature_store.cc
  minibatch_words.cc
  model.cc
  model_utils.cc
  ngram.cc
  ngram_filter.cc
  query_cache.cc
  parallel_processor.cc
  parallel_vocabulary.cc
  parallel_corpus.cc
  source_factored_weights.cc
  tree_metadata.cc
  utils.cc
  vocabulary.cc
  weights.cc
  word_context_hasher.cc
  word_distributions.cc
  word_to_class_index.cc
)

target_link_libs(lbl murmurhash rt ${Boost_LIBRARIES})

if(CDEC_FOUND)
  include_directories(cdec_ff_lbl PRIVATE ${CDEC_INCLUDE_DIRS})
  add_libraries(cdec_ff_lbl
      cdec_conditional_state_converter.cc
      cdec_ff_lbl.cc
      cdec_ff_source_lbl.cc
      cdec_lbl_mapper.cc
      cdec_rule_converter.cc
      cdec_state_converter.cc
      cdec_state_converter_base.cc
      create_ff.cc
      lbl_features.cc)

  target_link_libs(cdec_ff_lbl lbl)
endif(CDEC_FOUND)

#############################################
# Executables
#############################################

set(EXECUTABLES
  extract_word_vectors
  evaluate
  evaluate_parallel
  predict
  score
  score_parallel
  train_conditional_sgd
  train_factored_sgd
  train_maxent_sgd
  train_sgd
  train_tree_sgd
)
foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} lbl murmurhash utils)
endforeach(f)

add_subdirectory(tests)
