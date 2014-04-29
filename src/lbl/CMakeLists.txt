#############################################
# Libraries
#############################################

add_library(lbl SHARED
  circular_feature_batch.cc
  class_context_extractor.cc
  collision_global_feature_store.cc
  collision_minibatch_feature_store.cc
  collision_store.cc
  config.cc
  continuous_feature_batch.cc
  context_processor.cc
  factored_maxent_nlm.cc
  factored_nlm.cc
  feature_batch.cc
  feature_context.cc
  feature_context_extractor.cc
  feature_context_generator.cc
  feature_context_hasher.cc
  feature_context_keyer.cc
  feature_matcher.cc
  feature_store.cc
  feature_store_initializer.cc
  global_feature_indexes_pair.cc
  global_feature_store.cc
  minibatch_feature_indexes_pair.cc
  minibatch_feature_store.cc
  model_utils.cc
  nlm.cc
  nlm_approximate_z.cc
  sparse_global_feature_store.cc
  sparse_minibatch_feature_store.cc
  unconstrained_feature_store.cc
  utils.cc
  word_context_extractor.cc
  word_to_class_index.cc
)

if(CDEC_FOUND)
  include_directories(cdec_ff_lbl PRIVATE ${CDEC_INCLUDE_DIRS})
  add_library(cdec_ff_lbl SHARED cdec_ff_lbl.cc)
  target_link_libraries(cdec_ff_lbl lbl)
endif(CDEC_FOUND)

#############################################
# Executables
#############################################

set(EXECUTABLES train_factored_sgd train_maxent_sgd)
foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} lbl)
endforeach(f)

add_subdirectory(tests)
