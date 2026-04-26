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

# `grid-config --cxxflags` does not export the Grid source/build tree
# include paths or the libGrid.a location — automake adds those in-tree
# via `-I$(top_srcdir) -I$(top_builddir)`. Recover them by walking up
# from `grid-config`. Override `GRID_BUILD_ROOT` / `GRID_SRC_ROOT` if
# that heuristic ever fails.
GRID_CONFIG_DIR  := $(realpath $(dir $(GRID_CONFIG)))
GRID_BUILD_ROOT  ?= $(GRID_CONFIG_DIR)
GRID_SRC_ROOT    ?= $(realpath $(GRID_BUILD_ROOT)/..)
GRID_INCLUDES    := -I$(GRID_SRC_ROOT) -I$(GRID_BUILD_ROOT) -I$(GRID_BUILD_ROOT)/Grid
GRID_LIBDIR      := -L$(GRID_BUILD_ROOT)/Grid

SRC      := hmc_driver.cc
BINARIES := hmc_driver_leapfrog hmc_driver_omelyan hmc_driver_fg

all: $(BINARIES)

hmc_driver_leapfrog: $(SRC)
	$(CXX) $(GRID_INCLUDES) $(CXXFLAGS) -DHMC_INTEGRATOR=LeapFrog      -o $@ $< $(GRID_LIBDIR) $(LDFLAGS) $(LIBS)

hmc_driver_omelyan:  $(SRC)
	$(CXX) $(GRID_INCLUDES) $(CXXFLAGS) -DHMC_INTEGRATOR=MinimumNorm2  -o $@ $< $(GRID_LIBDIR) $(LDFLAGS) $(LIBS)

hmc_driver_fg:       $(SRC)
	$(CXX) $(GRID_INCLUDES) $(CXXFLAGS) -DHMC_INTEGRATOR=ForceGradient -o $@ $< $(GRID_LIBDIR) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(BINARIES)

.PHONY: all clean
