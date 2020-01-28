################################
# Makefile for Benkyokai2015
# by Makoto Shimazu <shimazu@eidos.ic.i.u-tokyo.ac.jp>
################################

PROJROOT := ..

include $(PROJROOT)/include.mk

# 測定用
VERSION   := 0.4.0
SRCDIR	  := src
INCDIR    := include
OBJDIR	  := obj
MLIB      := libmmmeasure.a
MSRCS	  := tester.cpp dataset.cpp resultstorage.cpp result.cpp
MOBJS	  := $(MSRCS:.cpp=.o)
MOBJS_FULL = $(addprefix $(OBJDIR)/, $(MOBJS))
MDEPS     := $(MSRCS:.cpp=.d)
MDEPS_FULL = $(addprefix $(OBJDIR)/, $(MDEPS))

ifeq ($(ELEM_TYPE),double)
CXXFLAGS += -DELEM_TYPE=double
endif
ifeq ($(ELEM_TYPE),float)
CXXFLAGS += -DELEM_TYPE=float
endif

.PHONY: all
all: lib

# lib -- library settings
.PHONY: lib
lib: $(MLIB)

$(MLIB): dirs $(MOBJS_FULL)
	$(AR) rc $@ $(MOBJS_FULL)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -DVERSION=\"$(VERSION)\" -DUSEMPI=$(USEMPI) -DPRINTWRONG=$(PRINTWRONG) -I$(INCDIR) -I$(SRCDIR) -c -o $@ $< $(LIBS)

.PHONY: dirs
dirs: $(OBJDIR)

$(OBJDIR):
	mkdir -pv $@

.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(MLIB)

-include $(MDEPS_FULL)
