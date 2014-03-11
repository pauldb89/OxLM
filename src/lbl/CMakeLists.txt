#############################################
# Libraries
#############################################

add_library(lbl SHARED nlm.cc feature.cc feature_generator.cc feature_store.cc)
if(CDEC_FOUND)
  include_directories(cdec_ff_lbl PRIVATE ${CDEC_INCLUDE_DIRS})
  add_library(cdec_ff_lbl SHARED cdec_ff_lbl.cc nlm.cc)

  add_library(cdec_ff_maxent SHARED cdec_ff_maxent.cc)
  target_link_libraries(cdec_ff_maxent lbl)
endif(CDEC_FOUND)

#############################################
# Executables
#############################################

set(EXECUTABLES train_factored_sgd train_maxent_sgd)
foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} lbl)
endforeach(f)
