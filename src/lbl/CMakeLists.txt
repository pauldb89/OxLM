#############################################
# Libraries
#############################################

add_library(lbl STATIC nlm.cc feature.cc feature_generator.cc feature_store.cc)

#############################################
# Executables
#############################################

set(EXECUTABLES train_factored_sgd train_maxent_sgd)
foreach(f ${EXECUTABLES})
  add_executable(${f} ${f}.cc)
  target_link_libraries(${f} lbl)
endforeach(f)
