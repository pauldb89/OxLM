#############################################
# Libraries
#############################################

# use SHARED after cnlm if *.so requried
# add_library(experiments SHARED paraphrase.cc paraphrase.h)

#############################################
# Executables
#############################################

set(EXECUTABLES pp-logprob pp-altprob)
foreach(f ${EXECUTABLES})
  add_executable(${f} ${f}.cc)
  target_link_libraries(${f} cnlm)
endforeach(f)
