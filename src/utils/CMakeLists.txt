# Produce a header file defining a constant with the current revision hash.
if(GIT_FOUND)
  include(GetGitRevisionDescription)
  get_git_head_revision(GIT_REFSPEC GIT_SHA1)
endif()
if(NOT GIT_FOUND OR GIT_SHA1 MATCHES ".*NOTFOUND.*")
  set(GIT_SHA1 "")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/git_revision.cc.in
    ${CMAKE_CURRENT_SOURCE_DIR}/git_revision.cc @ONLY)

add_library(utils SHARED git_revision.cc)
set_target_properties(utils PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(tests)
