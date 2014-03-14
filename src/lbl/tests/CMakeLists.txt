add_test(utils_test)

add_test(feature_test)
target_link_libraries(feature_test lbl)

add_test(feature_generator_test)
target_link_libraries(feature_generator_test lbl)

add_test(feature_store_test)
target_link_libraries(feature_store_test lbl)

add_test(train_factored_sgd_test)
target_link_libraries(train_factored_sgd_test lbl)

add_test(train_maxent_sgd_test)
target_link_libraries(train_maxent_sgd_test lbl)
