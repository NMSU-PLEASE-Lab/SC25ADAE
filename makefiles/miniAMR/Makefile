CC   = mpicc
LD   = mpicc
CPPFLAGS = -I.
CFLAGS = -O3 -fopenmp
LDFLAGS =
LDLIBS = -lm -fopenmp #-lgomp

#-----------------------------------------------------------------------
# APPEKG Section
# JEC: choose to build with APPEKG or not, and specify where
#      to find APPEKG stuff
APPEKGDIR = /anvil/projects/x-cis230165/tools/AppEKG/
DO_APPEKG = ON

ifeq ($(DO_APPEKG),ON)
APPEKGCFLAGS = -DAPPEKG -I${APPEKGDIR}
APPEKGLDOBJ = -L${APPEKGDIR} -lappekg -lpthread -lrt
else
APPEKGCFLAGS =
APPEKGLDOBJ =
endif

EXEC = miniAMRomp.x

OBJS = block.o check_sum.o comm_block.o comm.o comm_parent.o comm_refine.o \
       comm_util.o driver.o init.o main.o move.o pack.o plot.o profile.o \
       rcb.o refine.o stencil.o util.o

$(EXEC): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS) $(APPEKGLDOBJ)

.c.o:
	$(CC) $(CPPFLAGS) $(CFLAGS) $(APPEKGCFLAGS) -c $<

clean:
	rm *.o miniAMRomp.x

# need dependencies

block.o: block.h proto.h comm.h timer.h

check_sum.o: block.h comm.h proto.h

comm_block.o: block.h comm.h proto.h

comm.o: block.h comm.h proto.h timer.h

comm_parent.o: block.h comm.h proto.h

comm_refine.o: block.h comm.h proto.h

comm_util.o: block.h comm.h proto.h

driver.o: block.h comm.h proto.h timer.h

init.o: block.h proto.h comm.h

main.o: block.h comm.h proto.h param.h

move.o: block.h comm.h proto.h

pack.o: block.h comm.h proto.h

plot.o: block.h comm.h proto.h

profile.o: block.h comm.h proto.h timer.h

rcb.o: block.h comm.h proto.h timer.h

refine.o: block.h comm.h proto.h timer.h

stencil.o: block.h comm.h proto.h

util.o: block.h comm.h proto.h
