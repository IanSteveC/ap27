/*
	simpleCU.h

	CUDA backend for the AP26 GPU application.

	Drop-in replacement for simpleCL (the OpenCL helper used by the original
	AP26 app), reimplemented on the CUDA Driver API + NVRTC. It mirrors the
	subset of the simpleCL API that AP26.cpp / AP26.h actually use, so the host
	source stays almost unchanged (only its OpenCL device-init block becomes a
	CUDA one, and the cl_event wait helpers in AP26.h use CUevent).

	Two kernel-loading paths, selected at build time (same scheme as the
	genefer22 CUDA port):

	  * NVRTC (default)            - compile the embedded kernel source string at
	                                 runtime. Needs libnvrtc at runtime. Used for
	                                 development + bit-exact validation.

	  * Embedded fatbins           - define AP26_EMBED_FATBINS and provide
	    (-DAP26_EMBED_FATBINS)       fatbins.h with ap26_get_fatbin(). Loads a
	                                 pre-built multi-arch fatbin via the Driver
	                                 API (cuModuleLoadData); the driver selects
	                                 matching SASS. NO NVRTC at runtime - this is
	                                 the self-contained shipped BOINC binary.

	AP26 CUDA port.
*/

#ifndef SIMPLECU_H
#define SIMPLECU_H

#include <cuda.h>
#ifndef AP26_EMBED_FATBINS
#include <nvrtc.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#ifdef AP26_EMBED_FATBINS
/* Defined in fatbins.h, which AP26.cpp includes AFTER the kernel source-string
   headers (the registry keys on those source pointers). Returns NULL if none. */
const void * ap26_get_fatbin(const char * source);
#endif

/* ---------- OpenCL-compat shims -------------------------------------------
   The OpenCL scalar/handle types (cl_mem, cl_int, cl_uint, ...) are provided by
   BOINC's cl_boinc.h (pulled in by boinc_api.h, included before this header), so
   we must NOT redefine them. A cl_mem is treated as an opaque 8-byte handle in
   which we stash a CUdeviceptr; GPU events use CUevent directly. Only
   CL_MEM_READ_WRITE may be absent. */
#ifndef CL_MEM_READ_WRITE
#define CL_MEM_READ_WRITE 0            /* allocation mode is ignored under CUDA */
#endif

/* cl_mem <-> CUdeviceptr (both 8 bytes on LP64) */
static inline cl_mem      _cu_to_clmem(CUdeviceptr d) { return (cl_mem)(uintptr_t)d; }
static inline CUdeviceptr _cu_from_clmem(cl_mem m)    { return (CUdeviceptr)(uintptr_t)m; }

/* ---------- handles -------------------------------------------------------- */
typedef struct {
	CUdevice  device;
	CUcontext context;
	CUstream  queue;
	int       platform;                /* unused; kept for source compatibility */
} sclHard;

#define SCL_MAX_ARGS  16
#define SCL_MAX_ARGSZ 16
typedef struct {
	CUmodule      program;
	CUfunction    kernel;
	char          kernelName[256];
	size_t        global_size[3];
	size_t        local_size[3];
	/* CUDA has no persistent kernel-arg state, so cache args here for launch */
	unsigned char argData[SCL_MAX_ARGS][SCL_MAX_ARGSZ];
	int           nargs;
} sclSoft;

/* ---------- error helpers -------------------------------------------------- */
static inline void _sclCU(CUresult r, const char * what) {
	if (r != CUDA_SUCCESS) {
		const char * msg = 0;
		cuGetErrorString(r, &msg);
		printf("CUDA driver error (%s): %s\n", what, msg ? msg : "?");
		fprintf(stderr, "CUDA driver error (%s): %s\n", what, msg ? msg : "?");
		exit(EXIT_FAILURE);
	}
}

#ifndef AP26_EMBED_FATBINS
static inline void _sclNVRTC(nvrtcResult r, const char * what) {
	if (r != NVRTC_SUCCESS) {
		printf("NVRTC error (%s): %s\n", what, nvrtcGetErrorString(r));
		fprintf(stderr, "NVRTC error (%s): %s\n", what, nvrtcGetErrorString(r));
		exit(EXIT_FAILURE);
	}
}
#endif

/* ---------- hardware init / selection -------------------------------------- */
static inline sclHard sclGetCUDAHardware(int device_num) {
	sclHard h;
	memset(&h, 0, sizeof(h));

	_sclCU(cuInit(0), "cuInit");

	int ndev = 0;
	_sclCU(cuDeviceGetCount(&ndev), "cuDeviceGetCount");
	if (ndev < 1) {
		fprintf(stderr, "No CUDA devices found.\n");
		exit(EXIT_FAILURE);
	}
	if (device_num < 0 || device_num >= ndev) {
		fprintf(stderr, "Requested CUDA device %d out of range (have %d), using 0.\n", device_num, ndev);
		device_num = 0;
	}

	_sclCU(cuDeviceGet(&h.device, device_num), "cuDeviceGet");
	/* BLOCKING_SYNC: yield the CPU on synchronization (BOINC-friendly) */
	_sclCU(cuCtxCreate(&h.context, CU_CTX_SCHED_BLOCKING_SYNC, h.device), "cuCtxCreate");
	_sclCU(cuStreamCreate(&h.queue, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
	h.platform = 0;

	return h;
}

/* ---------- device queries ------------------------------------------------- */
static inline int _sclGetMaxComputeUnits(CUdevice device) {
	int v = 0;
	cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
	return v;
}
static inline unsigned long int _sclGetMaxMemAllocSize(CUdevice device) {
	size_t v = 0;
	cuDeviceTotalMem(&v, device);
	return (unsigned long int)v;
}
static inline unsigned long int _sclGetMaxGlobalMemSize(CUdevice device) {
	size_t v = 0;
	cuDeviceTotalMem(&v, device);
	return (unsigned long int)v;
}

/* ---------- buffers -------------------------------------------------------- */
static inline cl_mem sclMalloc(sclHard hardware, cl_int mode, size_t size) {
	(void)mode;
	CUdeviceptr d = 0;
	CUresult err = cuMemAlloc(&d, size);
	if (err != CUDA_SUCCESS) {
		const char * msg = 0; cuGetErrorString(err, &msg);
		printf("\nclMalloc (cuMemAlloc) Error: %s\n", msg ? msg : "?");
		fprintf(stderr, "CUDA memory allocation error, restarting in 1 minute.\n");
		boinc_temporary_exit(60);
	}
	return _cu_to_clmem(d);
}

static inline void sclWrite(sclHard hardware, size_t size, cl_mem buffer, void * hostPointer) {
	/* blocking + ordered on the work stream, like clEnqueueWriteBuffer(CL_TRUE) */
	_sclCU(cuMemcpyHtoDAsync(_cu_from_clmem(buffer), hostPointer, size, hardware.queue), "cuMemcpyHtoDAsync");
	_sclCU(cuStreamSynchronize(hardware.queue), "cuStreamSynchronize(write)");
}

static inline void sclRead(sclHard hardware, size_t size, cl_mem buffer, void * hostPointer) {
	/* blocking + ordered on the work stream, like clEnqueueReadBuffer(CL_TRUE) */
	_sclCU(cuMemcpyDtoHAsync(hostPointer, _cu_from_clmem(buffer), size, hardware.queue), "cuMemcpyDtoHAsync");
	_sclCU(cuStreamSynchronize(hardware.queue), "cuStreamSynchronize(read)");
}

static inline void sclReleaseMemObject(cl_mem object) {
	if (object) cuMemFree(_cu_from_clmem(object));
}

/* ---------- program build + kernel ---------------------------------------- */
static inline sclSoft sclGetCLSoftware(const char * source, const char * name, sclHard hardware, int opt) {
	(void)opt;
	sclSoft s;
	memset(&s, 0, sizeof(s));
	strncpy(s.kernelName, name, sizeof(s.kernelName) - 1);

#ifdef AP26_EMBED_FATBINS
	/* Shipped path: load the pre-built multi-arch fatbin for this kernel source.
	   The driver selects the SASS matching the runtime GPU. No NVRTC. */
	const void * fatbin = ap26_get_fatbin(source);
	if (fatbin == NULL) {
		fprintf(stderr, "No embedded fatbin for kernel '%s'.\n", name);
		exit(EXIT_FAILURE);
	}
	_sclCU(cuModuleLoadData(&s.program, fatbin), "cuModuleLoadData(fatbin)");
#else
	/* Dev/validation path: NVRTC-compile the embedded kernel source string. */
	int major = 0, minor = 0;
	cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hardware.device);
	cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hardware.device);
	char arch[64];
	snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major, minor);

	nvrtcProgram prog;
	_sclNVRTC(nvrtcCreateProgram(&prog, source, name, 0, NULL, NULL), "nvrtcCreateProgram");

	const char * opts[] = { arch };
	nvrtcResult cres = nvrtcCompileProgram(prog, 1, opts);
	if (cres != NVRTC_SUCCESS) {
		size_t logSize = 0;
		nvrtcGetProgramLogSize(prog, &logSize);
		char * log = (char *)malloc(logSize + 1);
		nvrtcGetProgramLog(prog, log);
		log[logSize] = 0;
		fprintf(stderr, "NVRTC compile failed for '%s':\n%s\n", name, log);
		printf("NVRTC compile failed for '%s':\n%s\n", name, log);
		free(log);
		exit(EXIT_FAILURE);
	}

	size_t ptxSize = 0;
	_sclNVRTC(nvrtcGetPTXSize(prog, &ptxSize), "nvrtcGetPTXSize");
	char * ptx = (char *)malloc(ptxSize);
	_sclNVRTC(nvrtcGetPTX(prog, ptx), "nvrtcGetPTX");
	nvrtcDestroyProgram(&prog);

	_sclCU(cuModuleLoadDataEx(&s.program, ptx, 0, NULL, NULL), "cuModuleLoadDataEx(ptx)");
	free(ptx);
#endif

	_sclCU(cuModuleGetFunction(&s.kernel, s.program, name), "cuModuleGetFunction");

	/* Mirror simpleCL: local size = the kernel's max work-group / block size. */
	int maxThreads = 0;
	cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, s.kernel);
	if (maxThreads < 1) maxThreads = 1;
	s.local_size[0]  = maxThreads; s.local_size[1]  = 1; s.local_size[2]  = 1;
	s.global_size[0] = maxThreads; s.global_size[1] = 1; s.global_size[2] = 1;

	printf("Kernel workgroup size: %d\n", maxThreads);
	return s;
}

/* Round the global size up to a whole number of blocks, then add one extra
   block - identical to simpleCL's sclSetGlobalSize. The kernels guard with
   `if (idx < N)`, so the surplus threads are harmless. */
static inline void sclSetGlobalSize(sclSoft & software, uint64_t size) {
	software.global_size[0] = ((size / software.local_size[0]) * software.local_size[0]) + software.local_size[0];
}

/* By reference (CUDA caches args in the sclSoft instead of on a kernel object).
   The host always passes a named global sclSoft, so this is a drop-in change. */
static inline void sclSetKernelArg(sclSoft & software, int argnum, size_t typeSize, void * argument) {
	if (argnum < 0 || argnum >= SCL_MAX_ARGS || typeSize > SCL_MAX_ARGSZ) {
		fprintf(stderr, "sclSetKernelArg: arg %d / size %zu out of range\n", argnum, typeSize);
		exit(EXIT_FAILURE);
	}
	memcpy(software.argData[argnum], argument, typeSize);
	if (argnum + 1 > software.nargs) software.nargs = argnum + 1;
}

static inline void _sclLaunch(sclHard hardware, sclSoft & software) {
	void * args[SCL_MAX_ARGS];
	for (int i = 0; i < software.nargs; i++) args[i] = (void *)software.argData[i];

	unsigned int bx = (unsigned int)software.local_size[0];
	unsigned int gx = (unsigned int)(software.global_size[0] / software.local_size[0]);
	_sclCU(cuLaunchKernel(software.kernel, gx, 1, 1, bx, 1, 1, 0, hardware.queue, args, NULL),
	       software.kernelName);
}

static inline void sclEnqueueKernel(sclHard hardware, sclSoft software) {
	_sclLaunch(hardware, software);
}

static inline CUevent sclEnqueueKernelEvent(sclHard hardware, sclSoft software) {
	_sclLaunch(hardware, software);
	CUevent e;
	cuEventCreate(&e, CU_EVENT_DEFAULT);
	cuEventRecord(e, hardware.queue);
	return e;
}

static inline double ProfilesclEnqueueKernel(sclHard hardware, sclSoft software) {
	CUevent start, stop;
	cuEventCreate(&start, CU_EVENT_DEFAULT);
	cuEventCreate(&stop,  CU_EVENT_DEFAULT);

	cuEventRecord(start, hardware.queue);
	_sclLaunch(hardware, software);
	cuEventRecord(stop, hardware.queue);
	cuEventSynchronize(stop);

	float ms = 0.0f;
	cuEventElapsedTime(&ms, start, stop);
	cuEventDestroy(start);
	cuEventDestroy(stop);
	return (double)ms;
}

static inline cl_int sclFinish(sclHard hardware) {
	_sclCU(cuStreamSynchronize(hardware.queue), "cuStreamSynchronize");
	return 0;
}

static inline void sclReleaseClSoft(sclSoft soft) {
	if (soft.program) cuModuleUnload(soft.program);
}

static inline void sclReleaseClHard(sclHard hardware) {
	if (hardware.queue)   cuStreamDestroy(hardware.queue);
	if (hardware.context) cuCtxDestroy(hardware.context);
}

#endif
