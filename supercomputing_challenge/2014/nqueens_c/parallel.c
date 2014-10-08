// Solve the NQueens problem for the <size> NxN board
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#define false 0
#define true 1
#define GRANULARITY 5
#define MASTER 0



int nprocs, myrank;


struct job {
	int work;				// true:work, false:quit
	int board[GRANULARITY];
};


struct job_msg {
	int solutions_found;
	int origin;
};



int master_place_queen(int column, int board[], int size) {
   	int n_solutions = 0;
	int is_sol;
	int i, j;

	for (i=0; i<size; i++) {
   		// Try to place a queen in each line of <column>
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
		  	if (column == GRANULARITY-1) {
				// If we are at the last level (granularity of the job),
				// This is a job for sending to a worker
				n_solutions += send_job_worker(board, size);
		   	} 
			else {
			   	// Not in the last level, try to place queens in the 
				// next one using the current board
				n_solutions += master_place_queen(column+1, board, size);
		   	}
	   	}
   	}

   	return n_solutions;
}



int worker_place_queen(int column, int board[], int size) {
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
				n_solutions += worker_place_queen(column+1, board, size);
		   	}
	   	}
   	}

   	return n_solutions;
}



int send_job_worker(int board[], int size) {
	int n_solutions = 0;		// The number of solutions found meanwhile
	int i;
	struct job todo;
	struct job_msg msg;
	
	// Build the job
	todo.work = true;

	for (i=0; i<GRANULARITY; i++)
		todo.board[i] = board[i];

	// Recieve the last result from a worker
	MPI_Recv(&msg, sizeof(msg), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, 
			 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	n_solutions = msg.solutions_found;

	// Send the new job to the worker
	MPI_Send(&todo, sizeof(todo), MPI_BYTE, msg.origin, 0, MPI_COMM_WORLD);

	return n_solutions;
}



int wait_remaining_results() {
	// Wait for remaining results, sending a quit whenever a new result arrives
	
	int n_solutions = 0;
	int n_workers = nprocs-1;
	struct job todo;
	struct job_msg msg;

	todo.work = false;

	while (n_workers > 0) {
		// Receive a message from a worker
		MPI_Recv(&msg, sizeof(msg), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, 
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		n_solutions += msg.solutions_found;

		MPI_Send(&todo, sizeof(todo), MPI_BYTE, msg.origin, 0, MPI_COMM_WORLD);

		n_workers -= 1;
	}

	return n_solutions;
	
}



void worker(int size) {
	// There is a default message named ask_job which lets a worker request a 
	// job reporting the number of solutions found in the last iteration
	
	int n_solutions;
	struct job_msg msg;

    msg.origin          = myrank;
	msg.solutions_found = 0;

	// Request initial job
	MPI_Send(&msg, sizeof(msg), MPI_BYTE, MASTER, 0, MPI_COMM_WORLD);

	while (true) {
		// Wait for a job or a quit message
	    struct job todo;
		MPI_Recv(&todo, sizeof(todo), MPI_BYTE, MPI_ANY_SOURCE, 
				 MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (todo.work == false)
		    break;

		n_solutions = worker_place_queen(GRANULARITY, todo.board, size);

		// Ask for more work
		msg.solutions_found = n_solutions;
		MPI_Send(&msg, sizeof(msg), MPI_BYTE, MASTER, 0, MPI_COMM_WORLD);
	}
}



void main(int argc, char* argv[]) {
	int n_solutions;
	int size=15;
	int board[size];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


	if (myrank == MASTER) {
		printf("size=%d\n", size);

		n_solutions = master_place_queen(0, board, size);
		n_solutions += wait_remaining_results();
		printf("n_solutions=%d\n", n_solutions);
	}
	else {
		worker(size);
	}

	MPI_Finalize();
}
