void calc_divv(int np, int nlev, int nelem, double *ru) {
	// size and intent of array arguments for f2py
	// ru :: np*np*(nlev+1)*nelem, inout
	
	int idx;

	for (idx=0; idx<np*np*(nlev+1)*nelem; idx++) {
		ru[idx] = 1.2;
	}
}
