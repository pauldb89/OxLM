# Libraries
add_library(cnlm SHARED cnlm.cc cnlm.h)

# Executables
add_executable(train_cnlm train_cnlm.cc)
target_link_libraries(train_cnlm ${Boost_LIBRARIES} cnlm)
