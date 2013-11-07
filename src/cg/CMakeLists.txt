#############################################
# Libraries
#############################################

# use SHARED after cnlm if *.so requried
add_library(cnlm SHARED cnlm.cc cnlm.h additive-cnlm.cc additive-cnlm.h)

#############################################
# Executables
#############################################

# add_executable(train_cnlm train_cnlm.cc cnlm.cc additive-cnlm.cc)
set(EXECUTABLES train_cnlm cg perplexity label printconfig)
foreach(f ${EXECUTABLES})
  add_executable(${f} ${f}.cc)
  target_link_libraries(${f} cnlm)
endforeach(f)
