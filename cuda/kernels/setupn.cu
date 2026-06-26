/*
	setup n59s on GPU kernel

*/


__constant__ unsigned long long MOD = (unsigned long long)258559632607830;


extern "C" __global__ void setupn(unsigned long long *n43_d, unsigned long long *n59_0, unsigned long long *n59_1, unsigned long long S53, unsigned long long S47, unsigned long long S43){

	int i = (blockIdx.x*blockDim.x + threadIdx.x);
	unsigned long long n43, n47, n53;
	int i43, i47, i53;
	int count;

	if(i<10840){

		n43 = n43_d[i];

		if(i<5420){
			count = i * 12673;
		}
		else{
			count = (i-5420) * 12673;
		}

		for(i43=19;i43>0;i43--){
			n47=n43;
			for(i47=23;i47>0;i47--){
				n53=n47;
				for(i53=29;i53>0;i53--){
					//n59=n53;
					if(i<5420){
						n59_0[count] = n53;
					}
					else{
						n59_1[count] = n53;
					}
					count++;

					n53+=S53;
					if(n53>=MOD)n53-=MOD;
				}
				n47+=S47;
				if(n47>=MOD)n47-=MOD;
			}
			n43+=S43;
			if(n43>=MOD)n43-=MOD;
		}

	}

}


