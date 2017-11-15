.PHONY : clean

CC = gcc
CFLAGS = -O3 -W -Wall

saxpy.so : saxpy.o
	$(CC) -shared -o saxpy.so saxpy.o

saxpy.o : saxpy.c
	$(CC) $(CFLAGS) -c -fPIC saxpy.c -o saxpy.o

clean :
	rm -f saxpy.so saxpy.o
