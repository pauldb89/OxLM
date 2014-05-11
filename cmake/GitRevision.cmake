execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
  OUTPUT_VARIABLE GIT_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE)
file(WRITE ../src/utils/git_revision.h
  "const char* GIT_REVISION = \"${GIT_REVISION}\";")
