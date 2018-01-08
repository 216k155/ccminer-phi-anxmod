/**
 * Lbry CUDA Implementation
 *
 * by tpruvot@github - July 2016
 *
 */

#include <string.h>
#include <stdint.h>

extern "C" {
#include <sph/sph_sha2.h>
#include <sph/sph_ripemd.h>
}

#include <cuda_helper.h>
#include <miner.h>

#define A 64
#define debug_cpu 0

extern "C" void lbry_hash(void* output, const void* input)
{
	uint32_t _ALIGN(A) hashA[16];
	uint32_t _ALIGN(A) hashB[8];
	uint32_t _ALIGN(A) hashC[8];

	sph_sha256_context ctx_sha256;
	sph_sha512_context ctx_sha512;
	sph_ripemd160_context ctx_ripemd;

	sph_sha256_init(&ctx_sha256);
	sph_sha256(&ctx_sha256, input, 112);
	sph_sha256_close(&ctx_sha256, hashA);

	sph_sha256(&ctx_sha256, hashA, 32);
	sph_sha256_close(&ctx_sha256, hashA);

	sph_sha512_init(&ctx_sha512);
	sph_sha512(&ctx_sha512, hashA, 32);
	sph_sha512_close(&ctx_sha512, hashA);

	sph_ripemd160_init(&ctx_ripemd);
	sph_ripemd160(&ctx_ripemd, hashA, 32);  // sha512 low
	sph_ripemd160_close(&ctx_ripemd, hashB);
	if (debug_cpu) applog_hex(hashB, 20);

	sph_ripemd160(&ctx_ripemd, &hashA[8], 32); // sha512 high
	sph_ripemd160_close(&ctx_ripemd, hashC);
	if (debug_cpu) applog_hex(hashC, 20);

	sph_sha256(&ctx_sha256, hashB, 20);
	sph_sha256(&ctx_sha256, hashC, 20);
	sph_sha256_close(&ctx_sha256, hashA);
	if (debug_cpu) applog_hex(hashA,32);

	sph_sha256(&ctx_sha256, hashA, 32);
	sph_sha256_close(&ctx_sha256, hashA);

	memcpy(output, hashA, 32);
}

/* ############################################################################################################################### */

extern void lbry_sha256_init(int thr_id);
extern void lbry_sha256_free(int thr_id);
extern void lbry_sha256_setBlock_112(uint32_t *pdata);
extern void lbry_sha256d_hash_112(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_outputHash);
extern void lbry_sha512_hash_32(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void lbry_sha256d_hash_final(int thr_id, uint32_t threads, uint32_t *d_inputHash, uint32_t *d_resNonce, const uint64_t target64);

extern void lbry_sha256_setBlock_112_merged(uint32_t *pdata);
extern void lbry_merged(int thr_id,uint32_t startNonce, uint32_t threads, uint32_t *d_resNonce, const uint64_t target64);

static __inline uint32_t swab32_if(uint32_t val, bool iftrue) {
	return iftrue ? swab32(val) : val;
}

static bool init[MAX_GPUS] = { 0 };

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
// nonce position is different
#define LBC_NONCE_OFT32 27

extern "C" int scanhash_lbry(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(A) vhash[8];
	uint32_t _ALIGN(A) endiandata[28];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[LBC_NONCE_OFT32];
	const int swap = 0; // to toggle nonce endian (need kernel change)

	const int dev_id = device_map[thr_id];
	uint32_t default_throughput = (device_sm[dev_id] > 500) ? ((1U<<26)+256*256*256) : ((1U<<23)+256*256*64);
	
	if (strstr(device_name[dev_id], "GTX 1080")) default_throughput=(1<<28);
	else 
	if (strstr(device_name[dev_id], "GTX 1070")) default_throughput=(1<<28);
	else 
	if (strstr(device_name[dev_id], "GTX 1060 6GB")) default_throughput=(1<<28);	
	else 
	if (strstr(device_name[dev_id], "GTX 980")) default_throughput =(1<<28);
	

	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark) {
		ptarget[7] = 0xf;
	}

	if (!init[thr_id]){
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);		

		if(device_sm[dev_id]<=500)
			CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 8 * sizeof(uint64_t) * throughput));

		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));
		
		CUDA_LOG_ERROR();

		init[thr_id] = true;
	}

	for (int i=0; i < LBC_NONCE_OFT32; i++) {
		be32enc(&endiandata[i], pdata[i]);
	}

	if(device_sm[dev_id]<=500)
		lbry_sha256_setBlock_112(endiandata);
	else
		lbry_sha256_setBlock_112_merged(endiandata);
	cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));
	do {
		// Hash with CUDA
		if(device_sm[dev_id]<=500){
			lbry_sha256d_hash_112(thr_id, throughput, pdata[LBC_NONCE_OFT32], d_hash[thr_id]);
			lbry_sha512_hash_32(thr_id, throughput, d_hash[thr_id]);
			lbry_sha256d_hash_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], *(uint64_t*)&ptarget[6]);
		}else{
			lbry_merged(thr_id,pdata[LBC_NONCE_OFT32], throughput, d_resNonce[thr_id], *(uint64_t*)&ptarget[6]);
		}
		uint32_t resNonces[2] = { UINT32_MAX, UINT32_MAX };
		cudaMemcpy(resNonces, d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	
		if (resNonces[ 0] != UINT32_MAX)
		{
			const uint32_t startNonce = pdata[LBC_NONCE_OFT32];
			
			resNonces[ 0]+= startNonce;
			
			endiandata[LBC_NONCE_OFT32] = swab32_if(resNonces[ 0], !swap);
			lbry_hash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				int res = 1;
				*hashes_done = pdata[LBC_NONCE_OFT32] - first_nonce + throughput;
				work->nonces[0] = swab32_if(resNonces[ 0], swap);
				work_set_target_ratio(work, vhash);
				if (resNonces[ 1] != UINT32_MAX) {
					resNonces[ 1]+=startNonce;
//					if(!opt_quiet)
//						gpulog(LOG_BLUE, thr_id, "Found 2nd nonce %08x", swab32(resNonces[ 1]));
					endiandata[LBC_NONCE_OFT32] = swab32_if(resNonces[ 1], !swap);
					lbry_hash(vhash, endiandata);
					work->nonces[1] = swab32_if(resNonces[ 1], swap);
					if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]) {
						work_set_target_ratio(work, vhash);
						xchg(work->nonces[0], work->nonces[1]);
					}
					res++;
				}
				pdata[LBC_NONCE_OFT32] = work->nonces[0];
				return res;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU %08x > %08x!", resNonces[ 0], vhash[7], ptarget[7]);
				cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));
			}
		}

		pdata[LBC_NONCE_OFT32] += throughput;

	} while (!work_restart[thr_id].restart && ((uint64_t)throughput + (uint64_t)pdata[LBC_NONCE_OFT32] < (uint64_t)max_nonce));

	*hashes_done = pdata[LBC_NONCE_OFT32] - first_nonce;

	return 0;
}

// cleanup
void free_lbry(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	if(device_sm[device_map[thr_id]]<=500)
		cudaFree(d_hash[thr_id]);
		
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
