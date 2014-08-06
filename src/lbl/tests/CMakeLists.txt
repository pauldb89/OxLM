set(TESTS
    bloom_filter_populator_test
    bloom_filter_test
    class_context_extractor_test
    class_context_hasher_test
    collision_counter_test
    collision_global_feature_store_test
    collision_minibatch_feature_store_test
    context_cache_test
    context_processor_test
    factored_metadata_test
    factored_weights_test
    feature_approximate_filter_test
    feature_context_test
    feature_context_generator_test
    feature_context_mapper_test
    feature_exact_filter_test
    feature_matcher_test
    feature_no_op_filter_test
    global_factored_maxent_weights_test
    global_feature_indexes_pair_test
    metadata_test
    minibatch_feature_indexes_pair_test
    model_test
    model_utils_test
    ngram_test
    ngram_filter_test
    operators_test
    query_cache_test
    sparse_global_feature_store_test
    sparse_minibatch_feature_store_test
    train_factored_sgd_test
    train_maxent_sgd_test
    train_sgd_test
    unconstrained_feature_store_test
    utils_test
    weights_test
    word_context_extractor_test
    word_context_hasher_test
    word_to_class_index_test)

set(LIBRARIES lbl)

if(CDEC_FOUND AND CDEC_LIBRARIES_FOUND)
  set(TESTS ${TESTS}
      cdec_lbl_mapper_test
      cdec_rule_converter_test
      cdec_state_converter_test
      lbl_features_test)
  set(LIBRARIES ${LIBRARIES} cdec_ff_lbl ${CDEC_LIBRARIES})
endif(CDEC_FOUND AND CDEC_LIBRARIES_FOUND)

foreach(test ${TESTS})
    lbl_add_test(${test})
    target_link_libraries(${test} ${LIBRARIES})
endforeach(test)
