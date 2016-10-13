void amb(int nx, double *a, double *b, double *c) {
	// size and intent of array arguments for f2py
	// a :: nx, in
	// b :: nx, in
	// c :: nx, inout
	
	int i;

	bmc(nx, LLL, b, c);

	for (i=0; i<nx; i++) {
		c[i] = KK*a[i] + c[i];
	}
}
