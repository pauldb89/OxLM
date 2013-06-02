LOCAL_LIBS=${HOME}/packages
CXXFLAGS =  -std=c++0x -g -O3 -fopenmp -Wall -I${LOCAL_LIBS}/include -m64 -I/usr/include/eigen3 -I..
LDFLAGS = -std=c++0x -g -O3 -fopenmp -L${LOCAL_LIBS}/lib64 -L${LOCAL_LIBS}/lib -lboost_program_options -lboost_serialization

all: hpyplm hpyplm_train hpyplm_query # dhpyplm chpyplm

hpyplm: hpyplm.o
	$(CXX) -o $@ $^ $(LDFLAGS)

#chpyplm: chpyplm.o
#	$(CXX) -o $@ $^ $(LDFLAGS)

#lbl_hpyplm: lbl_hpyplm.o ../lbl/log_bilinear_model.o
#	$(CXX) -o $@ $^ $(LDFLAGS)

#chpyplm_generate: chpyplm_generate.o
#	$(CXX) -o $@ $^ $(LDFLAGS)

#dhpyplm: dhpyplm.o
#	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	${CXX} $(CXXFLAGS) -c $< -o $@

%.o: %.c
	${CC} $(CFLAGS) -c $< -o $@


## stuff below here is optional

#BOOST_ROOT=$(LOCAL_LIBS)/boost_1_52_0/
BOOST_INCLUDE=$(LOCAL_LIBS)/include
#BOOST_SERIALIZATION=$(LOCAL_LIBS)/lib/libboost_serialization.a

hpyplm_train: hpyplm_train.o
	$(CXX) -o $@ $^ $(LDFLAGS)

hpyplm_query: hpyplm_query.o
	$(CXX) -o $@ $^ $(LDFLAGS)

hpyplm_query_observe: hpyplm_query_observe.cc
	$(CXX) -std=c++0x -O3 -Wall -I$(BOOST_INCLUDE) -I.. $< -o $@ $(BOOST_SERIALIZATION)


dhpyplm_train: dhpyplm_train.cc
	$(CXX) -std=c++0x -O3 -Wall -I$(BOOST_INCLUDE) -I.. $< -o $@ $(BOOST_SERIALIZATION)

dhpyplm_query: dhpyplm_query.cc
	$(CXX) -std=c++0x -O3 -Wall -I$(BOOST_INCLUDE) -I.. $< -o $@ $(BOOST_SERIALIZATION)

CDEC = ../../cdec
cdec_ff_hpyplm.o: cdec_ff_hpyplm.cc
	$(CXX) -shared -fPIC -std=c++0x -O3 -g -Wall -I$(BOOST_INCLUDE) -I.. -I$(CDEC)/utils -I$(CDEC)/mteval -I$(CDEC)/decoder $< -c

libcdec_ff_hpyplm.so: cdec_ff_hpyplm.o
	$(CXX) -shared -L../../cdec/decoder -lcdec -Wl,-soname,$@ -o $@ $<

.PHONY: depend echo
depend:
	$(CXX) -I.. -MM $(CXXFLAGS) hpyplm.{cc,h} dhpyplm.{cc,h} > makefile.depend

clean:
	rm -f *.o

clobber: clean
	rm makefile.depend hpyplm dhpyplm chpyplm # hpyplm_train hpyplm_query hpyplm_query_observe dhpyplm_train dhpyplm_query
