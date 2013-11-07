#############################################
# Libraries
#############################################

# use SHARED after cnlm if *.so requried
add_library(cnlm SHARED cnlm.cc cnlm.h)

#############################################
# Executables
#############################################

set(EXECUTABLES train_cnlm cg perplexity label printconfig)
foreach(f ${EXECUTABLES})
  add_executable(${f} ${f}.cc)
  target_link_libraries(${f} cnlm)
endforeach(f)
