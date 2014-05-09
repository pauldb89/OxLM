set(TESTS
    class_context_extractor_test
    class_hash_space_decider_test
    collision_counter_test
    collision_global_feature_store_test
    collision_minibatch_feature_store_test
    context_processor_test
    feature_context_test
    feature_context_generator_test
    feature_context_hasher_test
    feature_context_keyer_test
    feature_exact_filter_test
    feature_matcher_test
    feature_no_op_filter_test
    feature_store_initializer_test
    global_feature_indexes_pair_test
    minibatch_feature_indexes_pair_test
    model_utils_test
    ngram_query_test
    operators_test
    query_cache_test
    sparse_global_feature_store_test
    sparse_minibatch_feature_store_test
    train_maxent_sgd_test
    train_factored_sgd_test
    unconstrained_feature_store_test
    utils_test
    word_context_extractor_test
    word_to_class_index_test)

foreach(test ${TESTS})
    add_test(${test})
    target_link_libraries(${test} lbl)
endforeach(test)
