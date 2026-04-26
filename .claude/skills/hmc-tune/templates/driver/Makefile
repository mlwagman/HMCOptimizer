# Build three integrator variants of hmc_driver from one source.
#
# Drop this Makefile and hmc_driver.cc into a Grid build tree (next to
# tests/hmc, or anywhere grid-config is reachable) and invoke:
#
#   make -f Makefile GRID_CONFIG=/path/to/grid-config
#
# Produces:
#   hmc_driver_leapfrog    (LeapFrog)
#   hmc_driver_omelyan     (MinimumNorm2, a.k.a. Omelyan λ=0.1932)
#   hmc_driver_fg          (ForceGradient)

GRID_CONFIG ?= grid-config
CXX         := $(shell $(GRID_CONFIG) --cxx)
CXXFLAGS    := $(shell $(GRID_CONFIG) --cxxflags)
LDFLAGS     := $(shell $(GRID_CONFIG) --ldflags)
LIBS        := $(shell $(GRID_CONFIG) --libs)

SRC      := hmc_driver.cc
BINARIES := hmc_driver_leapfrog hmc_driver_omelyan hmc_driver_fg

all: $(BINARIES)

hmc_driver_leapfrog: $(SRC)
	$(CXX) $(CXXFLAGS) -DHMC_INTEGRATOR=LeapFrog      -o $@ $< $(LDFLAGS) $(LIBS)

hmc_driver_omelyan:  $(SRC)
	$(CXX) $(CXXFLAGS) -DHMC_INTEGRATOR=MinimumNorm2  -o $@ $< $(LDFLAGS) $(LIBS)

hmc_driver_fg:       $(SRC)
	$(CXX) $(CXXFLAGS) -DHMC_INTEGRATOR=ForceGradient -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(BINARIES)

.PHONY: all clean
