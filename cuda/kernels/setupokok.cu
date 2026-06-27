/*

	setupOKOK kernel

	Parallelized across (prime, residue) work-rows so the whole GPU is filled
	(was 542 threads / ~17 warps, badly load-imbalanced). Each thread computes
	exactly one OKOK word with an incremental residue (one modulo + 63 conditional
	subtracts instead of 64 modulos) and a single write-once store. Because every
	OKOK slot is written exactly once (the offset blocks are gapless and exactly
	prime-sized), no pre-zero is needed -> the clearokok kernel is dropped.

	Bit-identical to the original: residue r advances by step = MOD % P each jj
	(verified), and (i+(jj+shift)*MOD)%P == (base + jj*(MOD%P)) % P.
*/


__constant__ unsigned long long MOD = (unsigned long long)258559632607830;

__constant__ int g_primes[83] = {
	61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,
	163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,
	271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,
	397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,
	521,523,541 };

// one thread per (prime index, residue i); stride 542 = max(prime)+1
#define OKOK_STRIDE 542

extern "C" __global__ void setupokok(int shift, char *OK, unsigned long long *OKOK, int *offset){

	int tid  = (blockIdx.x*blockDim.x + threadIdx.x);
	int pidx = tid / OKOK_STRIDE;

	if(pidx < 83){
		int P = g_primes[pidx];
		int i = tid - pidx*OKOK_STRIDE;
		if(i < P){
			int o = offset[P];
			unsigned long long step = MOD % (unsigned long long)P;
			unsigned long long r = ((unsigned long long)i + (unsigned long long)shift * MOD) % (unsigned long long)P;

			unsigned long long word = 0;
			for(int jj=0; jj<64; jj++){
				word |= ((unsigned long long)OK[ (int)r + o ]) << jj;
				r += step;
				if(r >= (unsigned long long)P) r -= (unsigned long long)P;
			}

			OKOK[ o + i ] = word;
		}
	}
}
