#include <stdio.h>
#include <stdlib.h>


void print_matrix(int m, int **A) {
    int i, j;

    for (i=0; i<m; i++) {
        for (j=0; j<m; j++) {
            printf("%d\t", A[i][j]);
        }
        printf("\n");
    }
}


void doubling(int m, int **A) {
    int i, j;

    for (i=0; i<m; i++) {
        for (j=0; j<m; j++) {
            A[i][j] *= 2;
        }
    }
}


int myfunc(int i, int j, int **A) {
    return A[i+1][j+2] + 1;
}


void main(void) {
    int i, j;
    int m = 4;
    int **A;

    // allocation
    A = (int **) malloc(m * sizeof(int *));
    for (i=0; i<m; i++)
        A[i] = (int *) malloc(m * sizeof(int));

    // initialize
    int k = 0;
    for (i=0; i<m; i++) {
        for (j=0; j<m; j++) {
            A[i][j] = k;
            k++;
        }
    }

    // print the init values
    printf("(init) A = \n");
    print_matrix(m, A);

    // call the doubling function
    doubling(m, A);

    // print to verify
    printf("\n(doubling) A = \n");
    print_matrix(m, A);


    // call the myfunc
    printf("\n(myfunc) value = %d\n", myfunc(2, 1, A));
    
}
