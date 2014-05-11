# Produce a header file defining a constant with the current revision hash.
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/git_revision.cc.in
    ${CMAKE_CURRENT_SOURCE_DIR}/git_revision.cc @ONLY)

add_library(utils SHARED git_revision.cc)
set_target_properties(utils PROPERTIES LINKER_LANGUAGE CXX)
