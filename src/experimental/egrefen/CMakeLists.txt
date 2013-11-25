#############################################
# Libraries
#############################################

# use SHARED after cnlm if *.so requried
# add_library(experiments SHARED paraphrase.cc paraphrase.h)

#############################################
# Executables
#############################################

set(EXECUTABLES get_theta_l2norms)
foreach(f ${EXECUTABLES})
  add_executable(${f} ${f}.cc)
  target_link_libraries(${f} cnlm)
endforeach(f)
