/*
	checkn.cl
	tests primality of each term of the AP sequence
	test is good to 2^64-1

*/



// r0 + 2^64 * r1 = a * b
__device__ inline ulonglong2 mul_wide(const unsigned long long a, const unsigned long long b)
{
	ulonglong2 r;

#ifdef __NV_CL_C_VERSION
	const unsigned int a0 = (unsigned int)(a), a1 = (unsigned int)(a >> 32);
	const unsigned int b0 = (unsigned int)(b), b1 = (unsigned int)(b >> 32);

	unsigned int c0 = a0 * b0, c1 = __umulhi(a0, b0), c2, c3;

	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a0), "r" (b1), "r" (c1));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c2) : "r" (a0), "r" (b1));

	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b1), "r" (c2));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c3) : "r" (a1), "r" (b1));

	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a1), "r" (b0), "r" (c1));
	asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b0), "r" (c2));
	asm volatile ("addc.u32 %0, %1, 0;" : "=r" (c3) : "r" (c3));

	r.x = ((unsigned long long)c1 << 32) | c0; r.y = ((unsigned long long)c3 << 32) | c2;
#else
	r.x = a * b; r.y = __umul64hi(a, b);
#endif

	return r;
}


__device__ inline unsigned long long invert(unsigned long long p)
{
	unsigned long long p_inv = 1, prev = 0;
	while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }
	return p_inv;
}


__device__ inline unsigned long long montMul(unsigned long long a, unsigned long long b, unsigned long long p, unsigned long long q)
{
	ulonglong2 ab = mul_wide(a,b);

	unsigned long long m = ab.x * q;

	unsigned long long mp = __umul64hi(m,p);

	unsigned long long r = ab.y - mp;

	return ( ab.y < mp ) ? r + p : r;
}


__device__ inline unsigned long long add(unsigned long long a, unsigned long long b, unsigned long long p)
{
	unsigned long long r;

	unsigned long long c = (a >= p - b) ? p : 0;

	r = a + b - c;

	return r;
}

// strong probable prime to base 2
__device__ inline bool strong_prp(unsigned long long N)
{
	unsigned long long nmo = N-1;
	int t = 63 - __clzll(nmo & -nmo);	// this is ctz
	unsigned long long exp = N >> t;
	unsigned long long curBit = 0x8000000000000000;
	curBit >>= ( __clzll(exp) + 1 );

	unsigned long long q = invert(N);
	unsigned long long one = (-N) % N;
	unsigned long long a = add(one, one, N); 	// two, in montgomery form
	nmo = N - one;  		// N-1 in montgomery form

	/* If N is prime and N = d*2^t+1, where d is odd, then either
		1.  a^d = 1 (mod N), or
		2.  a^(d*2^s) = -1 (mod N) for some s in 0 <= s < t    */

  	/* r <-- a^d mod N, assuming d odd */
	while( curBit )
	{
		a = montMul(a,a,N,q);

		if(exp & curBit){
			a = add(a,a,N);
		}

		curBit >>= 1;
	}

	/* Clause 1. and s = 0 case for clause 2. */
	if (a == one || a == nmo){
		return true;
	}

	/* 0 < s < t cases for clause 2. */
	for (int s = 1; s < t; ++s){

		a = montMul(a,a,N,q);

		if(a == nmo){
	    		return true;
		}
	}


	return false;
}



/*
	main prime sequence checking kernel
*/
extern "C" __global__ void checkn(unsigned long long * n_result, unsigned long long STEP, int * sol_k, unsigned long long * sol_val, int * counter){

	int gid = (blockIdx.x*blockDim.x + threadIdx.x);

	if(gid < counter[0]){

		unsigned long long n = n_result[gid];

		unsigned long long m = n + STEP*5;

		if(m < n){  // software limit
			atomicOr(&counter[3], 1);
		}

		int k=0;

		// forward
		while(strong_prp( m )){
			m += STEP;
			++k;
			if(m < n){  // software limit
				atomicOr(&counter[3], 1);
				break;
			}
		}

		if(k >= 10){
			m = n + STEP*4;
			unsigned long long start = m;

			// reverse
			while(strong_prp( m )){
				m -= STEP;
				++k;
				if(m > start)break;  // m < 0
			}

			// AP length >= 10 store to results
			int index = atomicAdd(&counter[2], 1);
			sol_k[index] = k;
			sol_val[index] = m+STEP;
		}

	}

	// store largest ncount
	if(gid == 0){
		int nc = counter[0];
		if(nc > counter[1]){
			counter[1] = nc;
		}
	}
}
