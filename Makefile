WITH_EIGEN:=1
include $(BOB_ROBOTICS_PATH)/make_common/bob_robotics.mk

VECTOR_FIELD_SOURCES	:= vector_field.cc memory.cc
VECTOR_FIELD_OBJECTS	:= $(VECTOR_FIELD_SOURCES:.cc=.o)
VECTOR_FIELD_DEPS	:= $(VECTOR_FIELD_SOURCES:.cc=.d)

RIDF_SOURCES	:= ridf.cc memory.cc
RIDF_OBJECTS	:= $(RIDF_SOURCES:.cc=.o)
RIDF_DEPS	:= $(RIDF_SOURCES:.cc=.d)

CXXFLAGS +=-DENABLE_PREDEFINED_SOLID_ANGLE_UNITS
.PHONY: all clean

all: vector_field ridf

vector_field: $(VECTOR_FIELD_OBJECTS)
	$(CXX) -o $@ $(VECTOR_FIELD_OBJECTS) $(CXXFLAGS) $(LINK_FLAGS)

-include $(VECTOR_FIELD_DEPS)

ridf: $(RIDF_OBJECTS)
	$(CXX) -o $@ $(RIDF_OBJECTS) $(CXXFLAGS) $(LINK_FLAGS)

%.o: %.cc %.d
	$(CXX) -c -o $@ $< $(CXXFLAGS)
	
%.d: ;

clean:
	rm -f vector_field ridf *.d *.o
