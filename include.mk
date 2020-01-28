# 自分のつくったものに合うように変更すること
#YOURSRCS  := mymulmat.cpp main.cpp
YOURSRCS  := mymulmat.cpp cuda.cu main.cpp

# 要素の精度 (default: float)
#ELEM_TYPE := double
ELEM_TYPE := float

# XEONPHI/CUDA/MYLOCAL
#PLATFORM := MYLOCAL
PLATFORM := CUDA
# MPIを利用するときには1,そうでないときは0にする
USEMPI   := 0
# CUDAを利用するときには1,そうでないときは0にする
USECUDA  := 1
# Wrong!のプリントをするときには1, しないときには0
PRINTWRONG := 1

# 適宜変更しても大丈夫
ifeq ($(PLATFORM),XEONPHI)
CXX       = icpc
LD        = icpc
CXXFLAGS += -MMD -W -Wall -std=c++11 -axMIC-AVX512
LDFLAGS  +=
LIBS     +=
endif
ifeq ($(PLATFORM),CUDA)
CXX        = g++
NVCC       = nvcc
LD         = nvcc
CXXFLAGS  += -MMD -W -Wall -std=c++11
NVCCFLAGS += -std=c++11 -Xcompiler '-W -Wall' -arch=compute_60 -code=sm_60
LDFLAGS   += -arch=compute_60 -code=sm_60
LIBS      +=
endif
ifeq ($(PLATFORM),MYLOCAL)
CXX       = g++
LD        = g++
CXXFLAGS += -MMD -W -Wall -std=c++11
LDFLAGS  +=
LIBS     +=
endif
