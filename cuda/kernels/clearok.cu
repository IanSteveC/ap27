/*

	clearok kernel

		also clear counters

*/




extern "C" __global__ void clearok(char *OK, int *counter){


	int i = (blockIdx.x*blockDim.x + threadIdx.x);

	// clear array
	if(i < 23693){
		OK[i] = 1;
	}

	// clear counters
	if (i == 0){
		counter[1] = 0; // largest n count
		counter[2] = 0; // solutions
		counter[3] = 0; // PRP kernel overflow flag
	}

}




