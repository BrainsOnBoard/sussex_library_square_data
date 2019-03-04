WITH_EIGEN:=1
include $(BOB_ROBOTICS_PATH)/make_common/bob_robotics.mk


VECTOR_FIELD_SOURCES	:= vector_field.cc memory.cc
VECTOR_FIELD_OBJECTS	:= $(VECTOR_FIELD_SOURCES:.cc=.o)
VECTOR_FIELD_DEPS	:= $(VECTOR_FIELD_SOURCES:.cc=.d)

CXXFLAGS +=-DENABLE_PREDEFINED_SOLID_ANGLE_UNITS
.PHONY: all clean

all: vector_field

vector_field: $(VECTOR_FIELD_OBJECTS)
	$(CXX) -o $@ $(VECTOR_FIELD_OBJECTS) $(CXXFLAGS) $(LINK_FLAGS)

-include $(VECTOR_FIELD_DEPS)

%.o: %.cc %.d
	$(CXX) -c -o $@ $< $(CXXFLAGS)
	
%.d: ;

clean:
	rm -f vector_field *.d *.o
