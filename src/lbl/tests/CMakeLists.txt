set(TESTS
    class_context_extractor_test
    class_context_hasher_test
    class_distribution_test
    class_tree_test
    collision_counter_test
    collision_global_feature_store_test
    collision_minibatch_feature_store_test
    context_cache_test
    context_processor_test
    corpus_test
    factored_metadata_test
    factored_tree_weights_test
    factored_weights_test
    feature_context_test
    feature_context_generator_test
    feature_context_mapper_test
    feature_exact_filter_test
    feature_matcher_test
    global_factored_maxent_weights_test
    global_feature_indexes_pair_test
    metadata_test
    minibatch_words_test
    model_test
    model_utils_test
    ngram_test
    ngram_filter_test
    operators_test
    parallel_corpus_test
    parallel_processor_test
    parallel_vocabulary_test
    query_cache_test
    source_factored_weights_test
    train_conditional_sgd_test
    train_factored_sgd_test
    train_maxent_sgd_test
    train_sgd_test
    train_tree_sgd_test
    tree_metadata_test
    utils_test
    vocabulary_test
    weights_test
    word_context_extractor_test
    word_context_hasher_test
    word_distributions_test
    word_to_class_index_test)

set(LIBRARIES murmurhash lbl)

if(CDEC_FOUND AND CDEC_LIBRARIES_FOUND)
  set(TESTS ${TESTS}
      cdec_conditional_state_converter_test
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
