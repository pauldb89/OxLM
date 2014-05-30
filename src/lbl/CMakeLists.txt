#############################################
# Libraries
#############################################

add_library(lbl SHARED
  bloom_filter_populator.cc
  class_context_extractor.cc
  class_context_keyer.cc
  collision_counter.cc
  collision_global_feature_store.cc
  collision_minibatch_feature_store.cc
  collision_space.cc
  config.cc
  context_processor.cc
  factored_maxent_nlm.cc
  factored_nlm.cc
  feature_approximate_filter.cc
  feature_context.cc
  feature_context_extractor.cc
  feature_context_generator.cc
  feature_context_hasher.cc
  feature_context_keyer.cc
  feature_exact_filter.cc
  feature_filter.cc
  feature_no_op_filter.cc
  feature_matcher.cc
  feature_store.cc
  feature_store_initializer.cc
  global_feature_indexes_pair.cc
  global_feature_store.cc
  minibatch_feature_indexes_pair.cc
  minibatch_feature_store.cc
  model_utils.cc
  ngram_query.cc
  nlm.cc
  nlm_approximate_z.cc
  process_identifier.cc
  query_cache.cc
  sparse_global_feature_store.cc
  sparse_minibatch_feature_store.cc
  unconstrained_feature_store.cc
  utils.cc
  word_context_extractor.cc
  word_context_keyer.cc
  word_to_class_index.cc
)

target_link_libraries(lbl murmurhash)

if(CDEC_FOUND)
  include_directories(cdec_ff_lbl PRIVATE ${CDEC_INCLUDE_DIRS})
  add_library(cdec_ff_lbl SHARED cdec_ff_lbl.cc)
  target_link_libraries(cdec_ff_lbl lbl)
endif(CDEC_FOUND)

#############################################
# Executables
#############################################

set(EXECUTABLES train_factored_sgd train_maxent_sgd evaluate)
foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} lbl utils)
endforeach(f)

add_subdirectory(tests)
