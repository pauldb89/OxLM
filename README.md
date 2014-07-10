oxlm
====

Oxford language modelling code

**CODE HAS NOW MIGRATED TO BITBUCKET**

Run the following command in your git repository to update your remote:
`git remote set-url origin git@bitbucket.org:oxclg/oxlm.git` (if you have write
                                                              access).


**Installation**

We are migrating the code base to CMake. To install code, use the following
commands:


```
#!

mkdir build
cd build
cmake ../src
make
```

All binaries will automatically be placed in the bin/ directory within the main
project root, all libraries in the lib/ directory.

To run the unit tests coming with the LBL language modelling code, use the following commands:

```
#!

cd build
make all_tests
```
