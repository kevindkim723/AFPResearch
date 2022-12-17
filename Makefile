targets: afpgen.c afpgen.h
	gcc afpgen.c -o afpgen


clean:
	rm *.o afpgen