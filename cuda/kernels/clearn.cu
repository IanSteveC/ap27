/*

	clearn kernel

*/


extern "C" __global__ void clearn(int *counter){

	int i = (blockIdx.x*blockDim.x + threadIdx.x);

	if(i==0){
		counter[0]=0;
	}


}



