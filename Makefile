CC = mpicc
CFLAGS = -Wall -std=c99

all: qs_mpi

qs_mpi: qs_mpi.c
	$(CC) -o qs_mpi qs_mpi.c

clean:
	rm -f qs_mpi output.txt
