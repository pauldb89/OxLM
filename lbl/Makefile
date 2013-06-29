LOCAL_LIBS=${HOME}/packages
CXXFLAGS = -g -std=c++0x -fPIC -shared -O3 -fopenmp -Wall -I.. -I${LOCAL_LIBS}/include -m64 -I/usr/include/eigen3
LDFLAGS = -g -std=c++0x -O3 -fopenmp -L${LOCAL_LIBS}/lib64 -L${LOCAL_LIBS}/lib -lboost_program_options -lboost_serialization -lgfortran
FFLAGS = -Wall -g -O3  
CXX=g++
FC=gfortran

CXXFLAGS += -DEIGEN_DONT_PARALLELIZE

local_objs = log_bilinear_model.o lbfgs.o # lbfgs_wrapper.o

all: train_sgd train_factored_sgd

-include makefile.depend

#-----------------------#
# Local stuff
#-----------------------#

train_sgd: train_sgd.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

train_factored_sgd: train_factored_sgd.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

train_clbl: train_clbl.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

train_mlbl: train_mlbl.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

train_z: train_z.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	${CXX} $(CXXFLAGS) -c $< -o $@

%.o: %.c
	${CC} $(CFLAGS) -c $< -o $@
#
# %.o: %.f
#	$(FC) $(FFLAGS) -c $< -o $@
#
CDEC = ../../cdec
BOOST_INCLUDE=$(LOCAL_LIBS)/include
cdec_ff_lbl.o: cdec_ff_lbl.cc
	$(CXX) -shared -fPIC $(CXXFLAGS) -I$(CDEC)/utils -I$(CDEC)/mteval -I$(CDEC)/decoder $< -c

libcdec_ff_lbl.so: cdec_ff_lbl.o log_bilinear_model.o
	$(CXX) -shared -L../../cdec/decoder -lcdec -Wl,-soname,$@ -o $@ $^ $(LDFLAGS)

.PHONY: depend echo
depend:
	$(CXX) -MM $(CXXFLAGS) *.h *.cc > makefile.depend

clean:
	rm -f *.o

clobber: clean
	rm makefile.depend train_sgd libcdec_ff_lbl.so
