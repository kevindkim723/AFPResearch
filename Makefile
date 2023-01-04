targets: afpgen.c afpio.c afpgen.h
	gcc afpio.c -o afpio
#gcc  -c -o afpgen.o afpgen.c
#gcc  -c -o afpio.o afpio.c
#gcc  -o target afpio.o afpgen.o

clean:
	rm *.o afpgen