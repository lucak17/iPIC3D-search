#include "velocityHistogram.cuh"
#include "particleArraySoACUDA.cuh"

#include <iostream>
#include "cudaTypeDef.cuh"
#include "cudaReduction.cuh"
#include "dataAnalysisConfig.cuh"


namespace velocityHistogram
{

__global__ void histogramUpdateKernel(histogramTypeIn* minUVW, histogramTypeIn* maxUVW, int binDim0, int binDim1, velocityHistogramCUDA* histogramCUDAPtr){
    int idx = threadIdx.x;
    if(idx >= 3)return;

    histogramTypeIn min[2];
    histogramTypeIn max[2];
    int binDim[2] = {binDim0, binDim1};

    if(idx == 0){ // UV
        min[0] = minUVW[0];
        min[1] = minUVW[1];
        max[0] = maxUVW[0];
        max[1] = maxUVW[1];
    }else if(idx == 1){ // VW
        min[0] = minUVW[1];
        min[1] = minUVW[2];
        max[0] = maxUVW[1];
        max[1] = maxUVW[2];
    }else{ // UW
        min[0] = minUVW[0];
        min[1] = minUVW[2];
        max[0] = maxUVW[0];
        max[1] = maxUVW[2];
    }


    histogramCUDAPtr[idx].setHistogramDevice(min, max, binDim);


}


__global__ void velocityHistogramKernel(int nop, histogramTypeIn* u, histogramTypeIn* v, histogramTypeIn* w, histogramTypeIn* q,
                                        velocityHistogramCUDA* histogramCUDAPtr);

__global__ void resetBinScaleMarkKernel(velocityHistogramCUDA* histogramCUDAPtr);

__global__ void velocityHistogramKernelOne(int nop, histogramTypeIn* d1, histogramTypeIn* d2, histogramTypeIn* q,
                                        velocityHistogramCUDA* histogramCUDAPtr);


__host__ void velocityHistogram::init(velocitySoA* pclArray, int cycleNum, const int species, cudaStream_t stream){
    using namespace particleArraySoA;
    
    getRange(pclArray, species, stream);
    histogramUpdateKernel<<<1, 3, 0, stream>>>(reductionMinResultCUDA, reductionMaxResultCUDA, 
                                                binThisDim[0], binThisDim[1], 
                                                histogramCUDAPtr);

    // reset the histogram buffer, set the scalMark
    resetBinScaleMarkKernel<<<getGridSize(binThisDim[0] * binThisDim[1], 256), 256, 0, stream>>>(histogramCUDAPtr);

    // velocityHistogramKernel<<<getGridSize((int)pclArray->getNOP(), 256), 256, 0, stream>>>
    //     (pclArray->getNOP(), pclArray->getElement(U), pclArray->getElement(V), pclArray->getElement(W), pclArray->getElement(Q),
    //     histogramCUDAPtr);

    int sharedMemSize = sizeof(histogramTypeOut) * binThisDim[0] * binThisDim[1];

    velocityHistogramKernelOne<<<getGridSize((int)pclArray->getNOP() / 128, 512), 512, sharedMemSize, stream>>>
        (pclArray->getNOP(), pclArray->getElement(U), pclArray->getElement(V), pclArray->getElement(Q),
        histogramCUDAPtr);

    velocityHistogramKernelOne<<<getGridSize((int)pclArray->getNOP() / 128, 512), 512, sharedMemSize, stream>>>
        (pclArray->getNOP(), pclArray->getElement(V), pclArray->getElement(W), pclArray->getElement(Q),
        histogramCUDAPtr + 1);

    velocityHistogramKernelOne<<<getGridSize((int)pclArray->getNOP() / 128, 512), 512, sharedMemSize, stream>>>
        (pclArray->getNOP(), pclArray->getElement(U), pclArray->getElement(W), pclArray->getElement(Q),
        histogramCUDAPtr + 2);

    // copy the histogram object to host
    cudaErrChk(cudaMemcpyAsync(histogramHostPtr, histogramCUDAPtr, 3 * sizeof(velocityHistogramCUDA), cudaMemcpyDefault, stream));

    this->cycleNum = cycleNum;
}

__host__ int velocityHistogram::getRange(velocitySoA* pclArray, const int species, cudaStream_t stream){
    using namespace cudaReduction;

    constexpr int blockSize = 256;
    auto blockNum = reduceBlockNum(pclArray->getNOP(), blockSize);

    if(HISTOGRAM_FIXED_RANGE == false){
        for(int i=0; i<3; i++){ // UVW
            reduceMin<histogramTypeIn, blockSize><<<blockNum, blockSize, blockSize * sizeof(histogramTypeIn), stream>>>
                (pclArray->getElement(i), reductionTempArrayCUDA + i * reductionTempArraySize, pclArray->getNOP());
            reduceMinWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>
                (reductionTempArrayCUDA + i * reductionTempArraySize, reductionMinResultCUDA + i, blockNum);

            reduceMax<histogramTypeIn, blockSize><<<blockNum, blockSize, blockSize * sizeof(histogramTypeIn), stream>>>
                (pclArray->getElement(i), reductionTempArrayCUDA + (i+3) * reductionTempArraySize, pclArray->getNOP());
            reduceMaxWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>
                (reductionTempArrayCUDA + (i+3) * reductionTempArraySize, reductionMaxResultCUDA + i, blockNum);
        }
    }else{
        histogramTypeIn min = species == 0 || species == 2 ? -0.2 : -0.09;
        histogramTypeIn minArray[3] = {min, min, min};
        histogramTypeIn max = species == 0 || species == 2 ? 0.2 : 0.09;
        histogramTypeIn maxArray[3] = {max, max, max};

        cudaErrChk(cudaMemcpyAsync(reductionMinResultCUDA, minArray, 3 * sizeof(histogramTypeIn), cudaMemcpyDefault, stream));
        cudaErrChk(cudaMemcpyAsync(reductionMaxResultCUDA, maxArray, 3 * sizeof(histogramTypeIn), cudaMemcpyDefault, stream));
    }



    return 0;

}



}















