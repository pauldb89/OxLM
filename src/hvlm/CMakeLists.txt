#############################################
# Libraries
#############################################

# use SHARED after hvlm if *.so requried
add_library(hvlm SHARED hvlm.cc hvlm.h)

#############################################
# Executables
#############################################

# add_executable(train_cnlm train_cnlm.cc cnlm.cc additive-cnlm.cc)
set(EXECUTABLES )
foreach(f ${EXECUTABLES})
  add_executable(${f} ${f}.cc)
  target_link_libraries(${f} cnlm)
endforeach(f)
