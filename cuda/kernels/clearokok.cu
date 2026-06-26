/*

	clearokok kernel

*/


extern "C" __global__ void clearokok(unsigned long long *OKOK){


	int i = (blockIdx.x*blockDim.x + threadIdx.x);

	// clear array
	if(i < 23693){
		OKOK[i] = 0;
	}

}




