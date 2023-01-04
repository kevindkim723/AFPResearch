targets: afpgen.c afpio.c afpgen.h
	gcc afpgen.c -o afpgen
	gcc afpio.c -o afpio


clean:
	rm *.o afpgen