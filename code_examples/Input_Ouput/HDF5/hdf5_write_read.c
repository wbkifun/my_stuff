#include "hdf5.h"
#define FILE "file.h5"

int main() {
	hid_t	file_id;	// file identifier
	hid_t	dataspace_id, data
	herr_t	status;

	// Create a new file using default properties.
	file_id = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	// Create the data space for the dataset
	dims[0] = 4;
	dims[1] = 6;
	dataspace_id = H5Screate_simple(2, dims, NULL);
	
	// Terminate access to the file
	status = H5Fclose(file_id);

	return 0;
}
