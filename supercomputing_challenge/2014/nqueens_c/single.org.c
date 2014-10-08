// Solve the NQueens problem for the <size> NxN board
#include <stdio.h>
#define false 0
#define true 1



int n_queens(int size) {
    int board[size];
   	int n_solutions;
   
   	n_solutions = place_queen(0, board, size);

   	return n_solutions;
}



int place_queen(int column, int board[], int size) {
   	int n_solutions = 0;
	int is_sol;
	int i, j;

   	// Try to place a queen in each line of <column>
	for (i=0; i<size; i++) {
	   	board[column] = i;

	   	// Check if this board is still a solution
		is_sol = true;
		for (j=column-1; j>=0; j--) {
		   	if ((board[column] == board[j])               ||
				(board[column] == board[j] - (column-j))  ||
				(board[column] == board[j] + (column-j))) {
			   	is_sol = false;
			   	break;
		   	}
	   	}

	   	if (is_sol) {                    // It is a solution!
		  	if (column == size-1) {
			   	// If this is the last column, printout the solution
				n_solutions += 1;
			   	//print_solution(board, size);

		   	} else {
			   	// The board is not complete. Try to place the queens
				// on the next level, using the current board
				n_solutions+= place_queen(column+1, board, size);
		   	}
	   	}
   	}

   	return n_solutions;
}



int main() {
	int size, n_solutions;

	size = 18;
	n_solutions = n_queens(size);
	printf("size=%d\n", size);
	printf("n_solutions=%d\n", n_solutions);

	return 0;
}
