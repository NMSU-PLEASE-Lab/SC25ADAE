#-----------------------------------------------------------------------
# This file compiles for OpenMP and MPI hybrid operations using the GNU
# compile chain.

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=long

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX
#MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX


#-----------------------------------------------------------------------
# APPEKG Section
# JEC: choose to build with APPEKG or not, and specify where
#      to find APPEKG stuff
APPEKGDIR = /anvil/projects/x-cis230165/tools/ghappekg 
DO_APPEKG = ON

ifeq ($(DO_APPEKG),ON)
APPEKGCFLAGS = -DAPPEKG -I${APPEKGDIR}
APPEKGLDOBJ = -L${APPEKGDIR} -lappekg -lpthread -lrt
else
APPEKGCFLAGS =
APPEKGLDOBJ =
endif

#-----------------------------------------------------------------------

CFLAGS = -O3 -fopenmp
CXXFLAGS = $(CFLAGS)

CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) \
	$(MINIFE_MATRIX_TYPE) \
	-DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK \
	-DMINIFE_REPORT_RUSAGE ${APPEKGCFLAGS}

LDFLAGS=$(CFLAGS) ${APPEKGLDOBJ}
LIBS=

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

#CXX=mpiicpc
#CC=mpiicc

#CXX=g++
#CC=g++

#CXX=icpc
#CC=icc

CXX=mpicxx
CC=mpicc

include make_targets
