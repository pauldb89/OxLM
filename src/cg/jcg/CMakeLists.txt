#############################################
# Libraries
#############################################

# use SHARED after cnlm if *.so requried
add_library(gcnlm SHARED gcnlm.cc gcnlm.h additive-cnlm.cc additive-cnlm.h)

#############################################
# Executables
#############################################

set(EXECUTABLES train_jcg) # cg perplexity label printconfig)
foreach(f ${EXECUTABLES})
  add_executable(${f} ${f}.cc)
  target_link_libraries(${f} gcnlm)
endforeach(f)
