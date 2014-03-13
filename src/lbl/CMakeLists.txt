#############################################
# Libraries
#############################################

add_library(lbl SHARED
  config.cc
  context_extractor.cc
  factored_maxent_nlm.cc
  factored_nlm.cc
  feature_context.cc
  feature_generator.cc
  unconstrained_feature_store.cc
  nlm.cc
  nlm_approximate_z.cc
  utils.cc
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
