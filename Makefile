cc = mpicc 
CFLAGS = -O3 -funroll-loops -march=native -ffast-math
LIBS = -lm

BINS = matmul 

all: $(BINS)

matmul:main.c
	mpicc $(CFLAGS) -o matmul main.c $(LIBS)

clean:
	rm $(BINS)