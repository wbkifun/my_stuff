.PHONY : clean

CC = gcc
CFLAGS = -O3 -W -Wall

my_math.so : my_math.o
	$(CC) -shared -o my_math.so my_math.o

my_math.o : my_math.c
	$(CC) $(CFLAGS) -c -fPIC my_math.c -o my_math.o

clean :
	rm -f my_math.so my_math.o
