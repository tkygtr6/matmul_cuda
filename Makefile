################################
# Makefile for Benkyokai2015
# by Makoto Shimazu <shimazu@eidos.ic.i.u-tokyo.ac.jp>
################################

include ./include.mk

# 測定用
MAIN      := main.bin
SRCDIR	  := src
OBJDIR	  := obj
MMCOREDIR := mm-core
MINCDIR    := $(MMCOREDIR)/include
# Path to your codes
YOUROBJS      := $(YOURSRCS:.cpp=.o)
YOUROBJS      := $(YOUROBJS:.cu=.o)
YOUROBJS_FULL  = $(addprefix $(OBJDIR)/, $(YOUROBJS))
YOURDEPS      := $(YOURSRCS:.cpp=.d)
YOURDEPS      := $(YOURDEPS:.cu=.d)
YOURDEPS_FULL  = $(addprefix $(OBJDIR)/, $(YOURDEPS))

ifeq ($(ELEM_TYPE),double)
CXXFLAGS += -DELEM_TYPE=double
endif
ifeq ($(ELEM_TYPE),float)
CXXFLAGS += -DELEM_TYPE=float
endif

.PHONY: all
all: main

.PHONY: main
main: $(MAIN)

# main -- user build configurations
$(MAIN): dirs lib $(YOUROBJS_FULL)
	$(LD) $(LDFLAGS) -o $@ $(YOUROBJS_FULL) $(LIBS) -L$(MMCOREDIR) -lmmmeasure

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -DVERSION=\"$(VERSION)\" -DUSEMPI=$(USEMPI) -DPRINTWRONG=$(PRINTWRONG) -DUSECUDA=$(USECUDA) -I$(MINCDIR) -c -o $@ $< $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c -I$(MINCDIR) -o $@ $< $(LIBS)

# lib -- library settings
.PHONY: lib
lib:
	$(MAKE) -C $(MMCOREDIR)

.PHONY: dirs
dirs: $(OBJDIR)

$(OBJDIR):
	mkdir -pv $@

.PHONY: clean
clean:
	rm -rf $(MAIN) $(OBJDIR) $(YOUROBJDIR)

.PHONY: full_clean
full_clean: clean
	$(MAKE) -C $(MMCOREDIR) clean


-include $(YOURDEPS_FULL) $(MDEPS_FULL)
