LOCAL_LIBS=${HOME}/packages
CXXFLAGS = -g -std=c++0x -fPIC -shared -O3 -fopenmp -Wall -I.. -I${LOCAL_LIBS}/include -m64 -I/usr/include/eigen3
LDFLAGS = -g -std=c++0x -O3 -fopenmp -L${LOCAL_LIBS}/lib64 -L${LOCAL_LIBS}/lib -lboost_program_options -lboost_serialization
FFLAGS = -Wall -g -O3  
CXX=g++

CXXFLAGS += -DEIGEN_DONT_PARALLELIZE

local_objs = cnlm.o

all: train_cnlm cg

-include makefile.depend

#-----------------------#
# Local stuff
#-----------------------#

train_cnlm: train_cnlm.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

cg: cg.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	${CXX} $(CXXFLAGS) -c $< -o $@

%.o: %.c
	${CC} $(CFLAGS) -c $< -o $@

CDEC = ../../cdec
BOOST_INCLUDE=$(LOCAL_LIBS)/include
cdec_ff_lbl.o: cdec_ff_lbl.cc
	$(CXX) -shared -fPIC $(CXXFLAGS) -I$(CDEC)/utils -I$(CDEC)/mteval -I$(CDEC)/decoder $< -c

libcdec_ff_cnlm.so: cdec_ff_cnlm.o cnlm.o
	$(CXX) -shared -L../../cdec/decoder -lcdec -Wl,-soname,$@ -o $@ $^ $(LDFLAGS)

.PHONY: depend echo
depend:
	$(CXX) -MM $(CXXFLAGS) *.h *.cc > makefile.depend

clean:
	rm -f *.o

clobber: clean
	rm makefile.depend train_cnlm libcdec_ff_nlm.so libcdec_ff_cnlm.so
