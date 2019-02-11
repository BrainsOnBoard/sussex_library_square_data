WITH_EIGEN:=1
include $(BOB_ROBOTICS_PATH)/make_common/bob_robotics.mk

CXXFLAGS +=-DENABLE_PREDEFINED_SOLID_ANGLE_UNITS
.PHONY: all clean

all: vector_field

-include vector_field.d

vector_field: vector_field.cc vector_field.d
	$(CXX) -o $@ $< $(CXXFLAGS) $(LINK_FLAGS)

%.d: ;

clean:
	rm -f vector_field *.d
