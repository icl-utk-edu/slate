
ifeq ($(MAKECMDGOALS),mac)
	include mac.mk
else ifeq ($(MAKECMDGOALS),lin)
	include lin.mk
endif

mac lin:
	$(CC) $(CFLAGS) -I$(MPI)/include -DMPI -c trace/trace.c -o trace/trace.o
	$(CPP) $(CCFLAGS) $(INC) app.cc trace/trace.o $(LIB) -o app

clean:
	rm -rf app trace_*.svg
