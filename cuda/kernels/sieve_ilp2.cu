/*

	sieve kernel - 2-way ILP variant ("ilp2")

	Each thread runs TWO independent sito chains (adjacent n59 words 2t and
	2t+1) with per-word early-out guards and a joint dead-check. Doubles the
	memory-level parallelism per thread (10 batched shared loads before the
	first branch vs 5), hiding gather latency on GPUs whose occupancy is
	capped by the threads/SM ceiling (Turing = 1024, consumer Ampere/Ada/
	Blackwell = 1536). Bit-exact with sieve.cu: same words, same taps, same
	tail filter; only the atomicAdd result order changes (order-insensitive).

	HOST CONTRACT: `offset` is in THREAD units; a launch of G threads covers
	the 2G words [2*offset, 2*(offset+G)). The host p-loop must step p by
	global_size with bound halfn59s/2, and size n_result for 2x words
	(see sieve_wpt in AP26.cpp/AP26.h). halfn59s is even, so a pair never
	straddles the bound.

	fast 32 bit mod fails at approximately 2^54 which will never be reached
	because n59 does not exceed 2^48

*/

__constant__ int halfn59s = 68687660;
__constant__ unsigned long long MOD = (unsigned long long)258559632607830;

extern "C" __global__ void sieve(const unsigned long long * __restrict__ n59g, unsigned long long S59, int shift, unsigned long long * __restrict__ n_result, const unsigned long long * __restrict__ OKOK, int * __restrict__ counter, int offset){

	int idx = 2*((blockIdx.x*blockDim.x + threadIdx.x) + offset);

	__shared__ unsigned long long localOK[3198];
	// blockDim-agnostic copy (block size is tunable per arch via AP26_BLOCK)
	for(int q = threadIdx.x; q < 3198; q += blockDim.x){
		localOK[q] = OKOK[q];
	}
	__syncthreads();

	if(idx < halfn59s){
		unsigned long long n59A = n59g[idx];
		unsigned long long n59B = (idx+1 < halfn59s) ? n59g[idx+1] : 0;

		for(int i59=0;i59<35;i59++){
			unsigned int n59aA = n59A & ((1<<30)-1), n59bA = n59A >> 30;
			unsigned int n59aB = n59B & ((1<<30)-1), n59bB = n59B >> 30;
			unsigned long long sitoA, sitoB;

			sitoA = localOK[ ((n59aA+60*n59bA)%61) ]
				& localOK[ ((n59aA+25*n59bA)%67) + 61 ]
				& localOK[ ((n59aA+20*n59bA)%71) + 128 ]
				& localOK[ ((n59aA+8*n59bA)%73) + 199 ]
				& localOK[ ((n59aA+52*n59bA)%79) + 272 ];
			sitoB = localOK[ ((n59aB+60*n59bB)%61) ]
				& localOK[ ((n59aB+25*n59bB)%67) + 61 ]
				& localOK[ ((n59aB+20*n59bB)%71) + 128 ]
				& localOK[ ((n59aB+8*n59bB)%73) + 199 ]
				& localOK[ ((n59aB+52*n59bB)%79) + 272 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= localOK[ ((n59aA+40*n59bA)%83) + 351 ]
				& localOK[ ((n59aA+78*n59bA)%89) + 434 ]
				& localOK[ ((n59aA+33*n59bA)%97) + 523 ]
				& localOK[ ((n59aA+17*n59bA)%101) + 620 ]
				& localOK[ ((n59aA+93*n59bA)%103) + 721 ];
			if(sitoB) sitoB &= localOK[ ((n59aB+40*n59bB)%83) + 351 ]
				& localOK[ ((n59aB+78*n59bB)%89) + 434 ]
				& localOK[ ((n59aB+33*n59bB)%97) + 523 ]
				& localOK[ ((n59aB+17*n59bB)%101) + 620 ]
				& localOK[ ((n59aB+93*n59bB)%103) + 721 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= localOK[ ((n59aA+34*n59bA)%107) + 824 ]
				& localOK[ ((n59aA+46*n59bA)%109) + 931 ]
				& localOK[ ((n59aA+4*n59bA)%113) + 1040 ]
				& localOK[ ((n59aA+4*n59bA)%127) + 1153 ]
				& localOK[ ((n59aA+62*n59bA)%131) + 1280 ];
			if(sitoB) sitoB &= localOK[ ((n59aB+34*n59bB)%107) + 824 ]
				& localOK[ ((n59aB+46*n59bB)%109) + 931 ]
				& localOK[ ((n59aB+4*n59bB)%113) + 1040 ]
				& localOK[ ((n59aB+4*n59bB)%127) + 1153 ]
				& localOK[ ((n59aB+62*n59bB)%131) + 1280 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= localOK[ ((n59aA+77*n59bA)%137) + 1411 ]
				& localOK[ ((n59aA+45*n59bA)%139) + 1548 ]
				& localOK[ ((n59aA+144*n59bA)%149) + 1687 ]
				& localOK[ ((n59aA+n59bA)%151) + 1836 ];
			if(sitoB) sitoB &= localOK[ ((n59aB+77*n59bB)%137) + 1411 ]
				& localOK[ ((n59aB+45*n59bB)%139) + 1548 ]
				& localOK[ ((n59aB+144*n59bB)%149) + 1687 ]
				& localOK[ ((n59aB+n59bB)%151) + 1836 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= localOK[ ((n59aA+141*n59bA)%157) + 1987 ]
				& localOK[ ((n59aA+25*n59bA)%163) + 2144 ]
				& localOK[ ((n59aA+127*n59bA)%167) + 2307 ]
				& localOK[ ((n59aA+24*n59bA)%173) + 2474 ];
			if(sitoB) sitoB &= localOK[ ((n59aB+141*n59bB)%157) + 1987 ]
				& localOK[ ((n59aB+25*n59bB)%163) + 2144 ]
				& localOK[ ((n59aB+127*n59bB)%167) + 2307 ]
				& localOK[ ((n59aB+24*n59bB)%173) + 2474 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= localOK[ ((n59aA+121*n59bA)%179) + 2647 ]
				& localOK[ ((n59aA+49*n59bA)%181) + 2826 ];
			if(sitoB) sitoB &= localOK[ ((n59aB+121*n59bB)%179) + 2647 ]
				& localOK[ ((n59aB+49*n59bB)%181) + 2826 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= localOK[ ((n59aA+180*n59bA)%191) + 3007 ]
				& OKOK[ ((n59aA+27*n59bA)%193) + 3198 ];
			if(sitoB) sitoB &= localOK[ ((n59aB+180*n59bB)%191) + 3007 ]
				& OKOK[ ((n59aB+27*n59bB)%193) + 3198 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+22*n59bA)%197) + 3391 ]
				& OKOK[ ((n59aA+111*n59bA)%199) + 3588 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+22*n59bB)%197) + 3391 ]
				& OKOK[ ((n59aB+111*n59bB)%199) + 3588 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+171*n59bA)%211) + 3787 ]
				& OKOK[ ((n59aA+169*n59bA)%223) + 3998 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+171*n59bB)%211) + 3787 ]
				& OKOK[ ((n59aB+169*n59bB)%223) + 3998 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+44*n59bA)%227) + 4221 ]
				& OKOK[ ((n59aA+212*n59bA)%229) + 4448 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+44*n59bB)%227) + 4221 ]
				& OKOK[ ((n59aB+212*n59bB)%229) + 4448 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+2*n59bA)%233) + 4677 ]
				& OKOK[ ((n59aA+147*n59bA)%239) + 4910 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+2*n59bB)%233) + 4677 ]
				& OKOK[ ((n59aB+147*n59bB)%239) + 4910 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+64*n59bA)%241) + 5149 ]
				& OKOK[ ((n59aA+219*n59bA)%251) + 5390 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+64*n59bB)%241) + 5149 ]
				& OKOK[ ((n59aB+219*n59bB)%251) + 5390 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+193*n59bA)%257) + 5641 ]
				& OKOK[ ((n59aA+140*n59bA)%263) + 5898 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+193*n59bB)%257) + 5641 ]
				& OKOK[ ((n59aB+140*n59bB)%263) + 5898 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+79*n59bA)%269) + 6161 ]
				& OKOK[ ((n59aA+258*n59bA)%271) + 6430 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+79*n59bB)%269) + 6161 ]
				& OKOK[ ((n59aB+258*n59bB)%271) + 6430 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+76*n59bA)%277) + 6701 ]
				& OKOK[ ((n59aA+79*n59bA)%281) + 6978 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+76*n59bB)%277) + 6701 ]
				& OKOK[ ((n59aB+79*n59bB)%281) + 6978 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+204*n59bA)%283) + 7259 ]
				& OKOK[ ((n59aA+253*n59bA)%293) + 7542 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+204*n59bB)%283) + 7259 ]
				& OKOK[ ((n59aB+253*n59bB)%293) + 7542 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+114*n59bA)%307) + 7835 ]
				& OKOK[ ((n59aA+18*n59bA)%311) + 8142 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+114*n59bB)%307) + 7835 ]
				& OKOK[ ((n59aB+18*n59bB)%311) + 8142 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+19*n59bA)%313) + 8453 ]
				& OKOK[ ((n59aA+58*n59bA)%317) + 8766 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+19*n59bB)%313) + 8453 ]
				& OKOK[ ((n59aB+58*n59bB)%317) + 8766 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+n59bA)%331) + 9083 ]
				& OKOK[ ((n59aA+175*n59bA)%337) + 9414 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+n59bB)%331) + 9083 ]
				& OKOK[ ((n59aB+175*n59bB)%337) + 9414 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+292*n59bA)%347) + 9751 ]
				& OKOK[ ((n59aA+48*n59bA)%349) + 10098 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+292*n59bB)%347) + 9751 ]
				& OKOK[ ((n59aB+48*n59bB)%349) + 10098 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+191*n59bA)%353) + 10447 ]
				& OKOK[ ((n59aA+108*n59bA)%359) + 10800 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+191*n59bB)%353) + 10447 ]
				& OKOK[ ((n59aB+108*n59bB)%359) + 10800 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+15*n59bA)%367) + 11159 ]
				& OKOK[ ((n59aA+152*n59bA)%373) + 11526 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+15*n59bB)%367) + 11159 ]
				& OKOK[ ((n59aB+152*n59bB)%373) + 11526 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+335*n59bA)%379) + 11899 ]
				& OKOK[ ((n59aA+175*n59bA)%383) + 12278 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+335*n59bB)%379) + 11899 ]
				& OKOK[ ((n59aB+175*n59bB)%383) + 12278 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+295*n59bA)%389) + 12661 ]
				& OKOK[ ((n59aA+141*n59bA)%397) + 13050 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+295*n59bB)%389) + 12661 ]
				& OKOK[ ((n59aB+141*n59bB)%397) + 13050 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+164*n59bA)%401) + 13447 ]
				& OKOK[ ((n59aA+259*n59bA)%409) + 13848 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+164*n59bB)%401) + 13447 ]
				& OKOK[ ((n59aB+259*n59bB)%409) + 13848 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+273*n59bA)%419) + 14257 ]
				& OKOK[ ((n59aA+269*n59bA)%421) + 14676 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+273*n59bB)%419) + 14257 ]
				& OKOK[ ((n59aB+269*n59bB)%421) + 14676 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+144*n59bA)%431) + 15097 ]
				& OKOK[ ((n59aA+115*n59bA)%433) + 15528 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+144*n59bB)%431) + 15097 ]
				& OKOK[ ((n59aB+115*n59bB)%433) + 15528 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+65*n59bA)%439) + 15961 ]
				& OKOK[ ((n59aA+196*n59bA)%443) + 16400 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+65*n59bB)%439) + 15961 ]
				& OKOK[ ((n59aB+196*n59bB)%443) + 16400 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+81*n59bA)%449) + 16843 ]
				& OKOK[ ((n59aA+216*n59bA)%457) + 17292 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+81*n59bB)%449) + 16843 ]
				& OKOK[ ((n59aB+216*n59bB)%457) + 17292 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+447*n59bA)%461) + 17749 ]
				& OKOK[ ((n59aA+376*n59bA)%463) + 18210 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+447*n59bB)%461) + 17749 ]
				& OKOK[ ((n59aB+376*n59bB)%463) + 18210 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+13*n59bA)%467) + 18673 ]
				& OKOK[ ((n59aA+96*n59bA)%479) + 19140 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+13*n59bB)%467) + 18673 ]
				& OKOK[ ((n59aB+96*n59bB)%479) + 19140 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+328*n59bA)%487) + 19619 ]
				& OKOK[ ((n59aA+438*n59bA)%491) + 20106 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+328*n59bB)%487) + 19619 ]
				& OKOK[ ((n59aB+438*n59bB)%491) + 20106 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+111*n59bA)%499) + 20597 ]
				& OKOK[ ((n59aA+299*n59bA)%503) + 21096 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+111*n59bB)%499) + 20597 ]
				& OKOK[ ((n59aB+299*n59bB)%503) + 21096 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+216*n59bA)%509) + 21599 ]
				& OKOK[ ((n59aA+420*n59bA)%521) + 22108 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+216*n59bB)%509) + 21599 ]
				& OKOK[ ((n59aB+420*n59bB)%521) + 22108 ];
			if(!(sitoA|sitoB)) goto next;
			if(sitoA) sitoA &= OKOK[ ((n59aA+335*n59bA)%523) + 22629 ]
				& OKOK[ ((n59aA+189*n59bA)%541) + 23152 ];
			if(sitoB) sitoB &= OKOK[ ((n59aB+335*n59bB)%523) + 22629 ]
				& OKOK[ ((n59aB+189*n59bB)%541) + 23152 ];

			{ unsigned long long ss[2] = {sitoA, sitoB};
			  unsigned long long nn[2] = {n59A, n59B};
			  for(int w=0; w<2; w++){
				unsigned long long sito = ss[w];
				while(sito){
					int setbit = 63 - __clzll(sito);
					unsigned long long n = nn[w] + (unsigned long long)(setbit+shift)*MOD;
					if(n%7 && n%11 && n%13 && n%17 && n%19 && n%23){
						n_result[atomicAdd(&counter[0], 1)] = n;
					}
					sito ^= ((unsigned long long)1) << setbit;
				}
			  }
			}
next:
			n59A += S59; if(n59A >= MOD) n59A -= MOD;
			n59B += S59; if(n59B >= MOD) n59B -= MOD;
		}
	}
}
