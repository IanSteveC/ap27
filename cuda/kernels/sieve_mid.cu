/*

	sieve kernel - REDUCED shared cache variant ("mid")

	Identical to sieve.cu except only the primes <= 131 (15 primes, 1411
	entries = 11288 B) are cached in shared memory; 137..541 read global
	OKOK through L1. The small footprint lets TWO blocks fit the 32KB
	shared tier on 1536-thread/SM GPUs (consumer Ampere/Ada/Blackwell,
	block=768): 100% occupancy AND ~96KB L1 simultaneously - a combination
	the full 25.5KB cache cannot reach there. Bit-exact with sieve.cu.

	fast 32 bit mod fails at approximately 2^54 which will never be reached
	because n59 does not exceed 2^48

*/

__constant__ int halfn59s = 68687660;
__constant__ unsigned long long MOD = (unsigned long long)258559632607830;


extern "C" __global__ void sieve(const unsigned long long * __restrict__ n59g, unsigned long long S59, int shift, unsigned long long * __restrict__ n_result, const unsigned long long * __restrict__ OKOK, int * __restrict__ counter, int offset){

	int idx = (blockIdx.x*blockDim.x + threadIdx.x) + offset;

	__shared__ unsigned long long localOK[1411];
	// blockDim-agnostic copy (block size is tunable per arch via AP26_BLOCK)
	for(int q = threadIdx.x; q < 1411; q += blockDim.x){
		localOK[q] = OKOK[q];
	}
	__syncthreads();

	if(idx < halfn59s){

		unsigned long long sito;
		unsigned long long n59 = n59g[idx];

		int i59;
		for(i59=0;i59<35;i59++){
			unsigned int n59a = n59 & ((1<<30)-1);
			unsigned int n59b = n59 >> 30;

			if((sito  = localOK[ (n59a+60*n59b)%61 ]
				& localOK[ ((n59a+25*n59b)%67) + 61 ]
				& localOK[ ((n59a+20*n59b)%71) + 128 ]
				& localOK[ ((n59a+8*n59b)%73) + 199 ]
				& localOK[ ((n59a+52*n59b)%79) + 272 ]) )
			if(sito &= localOK[ ((n59a+40*n59b)%83) + 351 ]
				& localOK[ ((n59a+78*n59b)%89) + 434 ]
				& localOK[ ((n59a+33*n59b)%97) + 523 ]
				& localOK[ ((n59a+17*n59b)%101) + 620 ]
				& localOK[ ((n59a+93*n59b)%103) + 721 ] )
			if(sito &= localOK[ ((n59a+34*n59b)%107) + 824 ]
				& localOK[ ((n59a+46*n59b)%109) + 931 ]
				& localOK[ ((n59a+4*n59b)%113) + 1040 ]
				& localOK[ ((n59a+4*n59b)%127) + 1153 ]
				& localOK[ ((n59a+62*n59b)%131) + 1280 ] )
			if(sito &= OKOK[ ((n59a+77*n59b)%137) + 1411 ]
				& OKOK[ ((n59a+45*n59b)%139) + 1548 ]
				& OKOK[ ((n59a+144*n59b)%149) + 1687 ]
				& OKOK[ ((n59a+n59b)%151) + 1836 ] )
			if(sito &= OKOK[ ((n59a+141*n59b)%157) + 1987 ]
				& OKOK[ ((n59a+25*n59b)%163) + 2144 ]
				& OKOK[ ((n59a+127*n59b)%167) + 2307 ]
				& OKOK[ ((n59a+24*n59b)%173) + 2474 ] )
			if(sito &= OKOK[ ((n59a+121*n59b)%179) + 2647 ]
				& OKOK[ ((n59a+49*n59b)%181) + 2826 ])
			if(sito &= OKOK[ ((n59a+180*n59b)%191) + 3007 ]
				& OKOK[ ((n59a+27*n59b)%193) + 3198 ])
			if(sito &= OKOK[ ((n59a+22*n59b)%197) + 3391 ]
				& OKOK[ ((n59a+111*n59b)%199) + 3588 ])
			if(sito &= OKOK[ ((n59a+171*n59b)%211) + 3787 ]
				& OKOK[ ((n59a+169*n59b)%223) + 3998 ])
			if(sito &= OKOK[ ((n59a+44*n59b)%227) + 4221 ]
				& OKOK[ ((n59a+212*n59b)%229) + 4448 ])
			if(sito &= OKOK[ ((n59a+2*n59b)%233) + 4677 ]
				& OKOK[ ((n59a+147*n59b)%239) + 4910 ])
			if(sito &= OKOK[ ((n59a+64*n59b)%241) + 5149 ]
				& OKOK[ ((n59a+219*n59b)%251) + 5390 ])
			if(sito &= OKOK[ ((n59a+193*n59b)%257) + 5641 ]
				& OKOK[ ((n59a+140*n59b)%263) + 5898 ])
			if(sito &= OKOK[ ((n59a+79*n59b)%269) + 6161 ]
				& OKOK[ ((n59a+258*n59b)%271) + 6430 ])
			if(sito &= OKOK[ ((n59a+76*n59b)%277) + 6701 ]
				& OKOK[ ((n59a+79*n59b)%281) + 6978 ])
			if(sito &= OKOK[ ((n59a+204*n59b)%283) + 7259 ]
				& OKOK[ ((n59a+253*n59b)%293) + 7542 ])
			if(sito &= OKOK[ ((n59a+114*n59b)%307) + 7835 ]
				& OKOK[ ((n59a+18*n59b)%311) + 8142 ])
			if(sito &= OKOK[ ((n59a+19*n59b)%313) + 8453 ]
				& OKOK[ ((n59a+58*n59b)%317) + 8766 ])
			if(sito &= OKOK[ ((n59a+n59b)%331) + 9083 ]
				& OKOK[ ((n59a+175*n59b)%337) + 9414 ])
			if(sito &= OKOK[ ((n59a+292*n59b)%347) + 9751 ]
				& OKOK[ ((n59a+48*n59b)%349) + 10098 ])
			if(sito &= OKOK[ ((n59a+191*n59b)%353) + 10447 ]
				& OKOK[ ((n59a+108*n59b)%359) + 10800 ])
			if(sito &= OKOK[ ((n59a+15*n59b)%367) + 11159 ]
				& OKOK[ ((n59a+152*n59b)%373) + 11526 ])
			if(sito &= OKOK[ ((n59a+335*n59b)%379) + 11899 ]
				& OKOK[ ((n59a+175*n59b)%383) + 12278 ])
			if(sito &= OKOK[ ((n59a+295*n59b)%389) + 12661 ]
				& OKOK[ ((n59a+141*n59b)%397) + 13050 ])
			if(sito &= OKOK[ ((n59a+164*n59b)%401) + 13447 ]
				& OKOK[ ((n59a+259*n59b)%409) + 13848 ])
			if(sito &= OKOK[ ((n59a+273*n59b)%419) + 14257 ]
				& OKOK[ ((n59a+269*n59b)%421) + 14676 ])
			if(sito &= OKOK[ ((n59a+144*n59b)%431) + 15097 ]
				& OKOK[ ((n59a+115*n59b)%433) + 15528 ])
			if(sito &= OKOK[ ((n59a+65*n59b)%439) + 15961 ]
				& OKOK[ ((n59a+196*n59b)%443) + 16400 ])
			if(sito &= OKOK[ ((n59a+81*n59b)%449) + 16843 ]
				& OKOK[ ((n59a+216*n59b)%457) + 17292 ])
			if(sito &= OKOK[ ((n59a+447*n59b)%461) + 17749 ]
				& OKOK[ ((n59a+376*n59b)%463) + 18210 ])
			if(sito &= OKOK[ ((n59a+13*n59b)%467) + 18673 ]
				& OKOK[ ((n59a+96*n59b)%479) + 19140 ])
			if(sito &= OKOK[ ((n59a+328*n59b)%487) + 19619 ]
				& OKOK[ ((n59a+438*n59b)%491) + 20106 ])
			if(sito &= OKOK[ ((n59a+111*n59b)%499) + 20597 ]
				& OKOK[ ((n59a+299*n59b)%503) + 21096 ])
			if(sito &= OKOK[ ((n59a+216*n59b)%509) + 21599 ]
				& OKOK[ ((n59a+420*n59b)%521) + 22108 ])
			if(sito &= OKOK[ ((n59a+335*n59b)%523) + 22629 ]
				& OKOK[ ((n59a+189*n59b)%541) + 23152 ]){


				if(__popcll(sito) == 1){
					int setbit = 63 - __clzll(sito);
					unsigned long long n=n59+(setbit+shift)*MOD;

					if(n%7 && n%11 && n%13 && n%17 && n%19 && n%23){
						n_result[atomicAdd(&counter[0], 1)] = n;
					}
				}
				else{  // more than 1 set bit
					do{
						int setbit = 63 - __clzll(sito);
						unsigned long long n=n59+(setbit+shift)*MOD;

						if(n%7 && n%11 && n%13 && n%17 && n%19 && n%23){
							n_result[atomicAdd(&counter[0], 1)] = n;
						}

						sito ^= ((unsigned long long)1) << setbit; // toggle bit off

					} while(sito);
				}
			}


			n59+=S59;
			if(n59>= MOD ){
				n59-= MOD;
			}

		}

	}

}
