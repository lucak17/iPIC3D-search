
#include <thread>
#include <future>
#include <string>
#include <memory>
#include <random>

#include "iPic3D.h"
#include "VCtopology3D.h"
#include "outputPrepare.h"
#include "threadPool.hpp"

#include "dataAnalysis.cuh"
#include "dataAnalysisConfig.cuh"
#include "GMM/cudaGMM.cuh"
#include "particleArraySoACUDA.cuh"
#include "velocityHistogram.cuh"



namespace dataAnalysis
{

using namespace iPic3D;
using velocitySoA = particleArraySoA::particleArraySoACUDA<cudaCommonType, 0, 3>;


class dataAnalysisPipelineImpl {
using weightType = cudaTypeSingle;
private:
    int ns;
    int deviceOnNode;
    // pointers to objects in KCode
    cudaStream_t* streams;
    particleArrayCUDA** pclsArrayHostPtr = nullptr;

    std::future<int> analysisFuture;

    ThreadPool* DAthreadPool = nullptr;
    velocitySoA* velocitySoACUDA = nullptr;
    // histogram
    string HistogramSubDomainOutputPath;
    velocityHistogram::velocityHistogram* velocityHistogram = nullptr;

    // GMM
    string GMMSubDomainOutputPath;
    cudaGMMWeight::GMM<cudaCommonType, 2, weightType>* gmmArray = nullptr;

public:

    dataAnalysisPipelineImpl(c_Solver& KCode) {
        ns = KCode.ns;
        deviceOnNode = KCode.cudaDeviceOnNode;
        streams = KCode.streams;
        pclsArrayHostPtr = KCode.pclsArrayHostPtr;

        DAthreadPool = new ThreadPool(4);

        if constexpr (VELOCITY_HISTOGRAM_ENABLE) { // velocity histogram
            velocitySoACUDA = new velocitySoA();

            HistogramSubDomainOutputPath = HISTOGRAM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "/";
            velocityHistogram = new velocityHistogram::velocityHistogram(12000);

            if constexpr (GMM_ENABLE) { // GMM
                GMMSubDomainOutputPath = GMM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "/";
                gmmArray = new cudaGMMWeight::GMM<cudaCommonType, 2, weightType>[3];
            }
        }
    }

    void startAnalysis(int cycle);

    int checkAnalysis();

    int waitForAnalysis();


    ~dataAnalysisPipelineImpl() {
        if (DAthreadPool != nullptr) delete DAthreadPool;
        if (velocitySoACUDA != nullptr) delete velocitySoACUDA;
        if (velocityHistogram != nullptr) delete velocityHistogram;
        if (gmmArray != nullptr) delete[] gmmArray;
    }

private:

    int analysisEntre(int cycle);

    int GMMAnalysisSpecies(int cycle, std::string outputPath);
    
};



/**
 * @brief analysis function for each species, uv, uw, vw
 * @details It launches 3 threads for uv uw vw analysis in parallel
 * 
 */
int dataAnalysisPipelineImpl::GMMAnalysisSpecies(int cycle, std::string outputPath){

    using weightType = cudaTypeSingle;

    std::future<int> future[3];

    auto GMMLambda = [=](int i) mutable {

        using namespace cudaGMMWeight;

        std::random_device rd;  // True random seed (if available)
        std::mt19937 gen(rd()); // Mersenne Twister PRNG (better than rand())
        std::uniform_real_distribution<cudaCommonType> dist(-0.001, 0.001); // Range [0,2]

        cudaErrChk(cudaSetDevice(deviceOnNode));

        // GMM config
        constexpr auto numComponent = 8;
        // constexpr double uniformweight = 1/numComponent;

        cudaCommonType weightVector[numComponent];
        cudaCommonType meanVector[numComponent * 2];
        cudaCommonType coVarianceMatrix[numComponent * 4];

        for(int j = 0; j < numComponent; j++){
            weightVector[j] = 1.0/numComponent;
            meanVector[j * 2] = dist(gen);
            meanVector[j * 2 + 1] = dist(gen);
            coVarianceMatrix[j * 4] = 0.001;
            coVarianceMatrix[j * 4 + 1] = 0.0;
            coVarianceMatrix[j * 4 + 2] = 0.0;
            coVarianceMatrix[j * 4 + 3] = 0.001;
        }

        GMMParam_t<cudaCommonType> GMMParam = {
            .numComponents = numComponent,
            .maxIteration = 50,
            .threshold = 1e-6,

            .weightInit = weightVector,
            .meanInit = meanVector,
            .coVarianceInit = coVarianceMatrix
        };

        // data
        GMMDataMultiDim<cudaCommonType, 2, weightType> GMMData
            (10000, velocityHistogram->getHistogramScaleMark(i), velocityHistogram->getVelocityHistogramCUDAArray(i));

        cudaErrChk(cudaHostRegister(&GMMData, sizeof(GMMData), cudaHostRegisterDefault));
        
        // generate exact output file path
        std::string uvw[3] = {"/uv_", "/uw_", "/vw_"};
        auto fileOutputPath = outputPath + uvw[i] + std::to_string(cycle) + ".json";

        auto& gmm = gmmArray[i];
        gmm.config(&GMMParam, &GMMData);
        auto convergStep = gmm.initGMM(); // the exact output file name
        int ret = 0;
        if constexpr (GMM_OUTPUT) ret = gmm.outputGMM(convergStep, fileOutputPath);

        cudaErrChk(cudaHostUnregister(&GMMData));
        
        return ret;
    };

    for(int i = 0; i < 3; i++){
        // launch 3 async threads for uv, uw, vw
        future[i] = DAthreadPool->enqueue(GMMLambda, i); 
    }

    for(int i = 0; i < 3; i++){
        future[i].wait();
    }

    return 0;
}

/**
 * @brief analysis function, called by startAnalysis
 * @details procesures in this function should be executed in sequence, the order of the analysis should be defined here
 *          But the procedures can launch other threads to do the analysis
 *          Also this function is a friend function of c_Solver, resources in the c_Slover should be dispatched here
 */
int dataAnalysisPipelineImpl::analysisEntre(int cycle){
    cudaErrChk(cudaSetDevice(deviceOnNode));

    // species by species to save VRAM
    for(int i = 0; i < ns; i++){
        if constexpr (VELOCITY_HISTOGRAM_ENABLE) {
            // to SoA
            velocitySoACUDA->updateFromAoS(pclsArrayHostPtr[i], streams[i]);

            // histogram
            auto histogramSpeciesOutputPath = HistogramSubDomainOutputPath + "species" + std::to_string(i) + "/";
            velocityHistogram->init(velocitySoACUDA, cycle, i, streams[i]);
            if constexpr (HISTOGRAM_OUTPUT)
            velocityHistogram->writeToFileFloat(histogramSpeciesOutputPath, streams[i]); // TODO
            else cudaErrChk(cudaStreamSynchronize(streams[i]));

            if constexpr (GMM_ENABLE) { // GMM
                auto GMMSpeciesOutputPath = GMMSubDomainOutputPath + "species" + std::to_string(i) + "/";
                GMMAnalysisSpecies(cycle, GMMSpeciesOutputPath);
            }
        }
    }

    return 0;
}


/**
 * @brief start all the analysis registered here
 */
void dataAnalysisPipelineImpl::startAnalysis(int cycle){

    if(DATA_ANALYSIS_EVERY_CYCLE == 0 || (cycle % DATA_ANALYSIS_EVERY_CYCLE != 0)){
        analysisFuture = std::future<int>();
    } else {
        analysisFuture = DAthreadPool->enqueue(&dataAnalysisPipelineImpl::analysisEntre, this, cycle); 

        if(analysisFuture.valid() == false){
            throw std::runtime_error("[!]Error: Can not start data analysis");
        }
    }

}

/**
 * @brief check if the analysis is done, non-blocking
 * 
 * @return 0 if the analysis is done, 1 if it is not done
 */
int dataAnalysisPipelineImpl::checkAnalysis(){

    if(analysisFuture.valid() == false){
        return 0;
    }

    if(analysisFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
        return 0;
    }else{
        return 1;
    }

    return 0;
}

/**
 * @brief wait for the analysis to be done, blocking
 */
int dataAnalysisPipelineImpl::waitForAnalysis(){

    if(analysisFuture.valid() == false){
        return 0;
    }

    analysisFuture.wait();

    return 0;
}


/**
 * @brief create output directory for the data analysis, controlled by dataAnalysisConfig.cuh
 */
void dataAnalysisPipeline::createOutputDirectory(int myrank, int ns, VirtualTopology3D* vct){ // output path for data analysis
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return;
    }

    // VCT mapping for this subdomain
    auto writeVctMapping = [&](const std::string& filePath) {
        std::ofstream vctMapping(filePath);
        if(vctMapping.is_open()){
        vctMapping << "Cartesian rank: " << vct->getCartesian_rank() << std::endl;
        vctMapping << "Number of processes: " << vct->getNprocs() << std::endl;
        vctMapping << "XLEN: " << vct->getXLEN() << std::endl;
        vctMapping << "YLEN: " << vct->getYLEN() << std::endl;
        vctMapping << "ZLEN: " << vct->getZLEN() << std::endl;
        vctMapping << "X: " << vct->getCoordinates(0) << std::endl;
        vctMapping << "Y: " << vct->getCoordinates(1) << std::endl;
        vctMapping << "Z: " << vct->getCoordinates(2) << std::endl;
        vctMapping << "PERIODICX: " << vct->getPERIODICX() << std::endl;
        vctMapping << "PERIODICY: " << vct->getPERIODICY() << std::endl;
        vctMapping << "PERIODICZ: " << vct->getPERIODICZ() << std::endl;

        vctMapping << "Neighbor X left: " << vct->getXleft_neighbor() << std::endl;
        vctMapping << "Neighbor X right: " << vct->getXright_neighbor() << std::endl;
        vctMapping << "Neighbor Y left: " << vct->getYleft_neighbor() << std::endl;
        vctMapping << "Neighbor Y right: " << vct->getYright_neighbor() << std::endl;
        vctMapping << "Neighbor Z left: " << vct->getZleft_neighbor() << std::endl;
        vctMapping << "Neighbor Z right: " << vct->getZright_neighbor() << std::endl;

        vctMapping.close();
        } else {
        throw std::runtime_error("[!]Error: Can not create VCT mapping for velocity GMM species");
        }
    };

    if constexpr (VELOCITY_HISTOGRAM_ENABLE && HISTOGRAM_OUTPUT) {
        auto histogramSubDomainOutputPath = HISTOGRAM_OUTPUT_DIR + "subDomain" + std::to_string(myrank) + "/";
        for(int i = 0; i < ns; i++){
            auto histogramSpeciesOutputPath = histogramSubDomainOutputPath + "species" + std::to_string(i);
            if(0 != checkOutputFolder(histogramSpeciesOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity histogram species");
            }
        }
        writeVctMapping(histogramSubDomainOutputPath + "vctMapping.txt");
    }

    if constexpr (GMM_ENABLE && GMM_OUTPUT) {
        auto GMMSubDomainOutputPath = GMM_OUTPUT_DIR + "subDomain" + std::to_string(myrank) + "/";
        for(int i = 0; i < ns; i++){
            auto GMMSpeciesOutputPath = GMMSubDomainOutputPath + "species" + std::to_string(i) + "/";
            if(0 != checkOutputFolder(GMMSpeciesOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity GMM species");
            }
        }
        writeVctMapping(GMMSubDomainOutputPath + "vctMapping.txt");
    }

}



dataAnalysisPipeline::dataAnalysisPipeline(iPic3D::c_Solver& KCode) {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        impl = nullptr;
        return;
    }
    impl = std::make_unique<dataAnalysisPipelineImpl>(std::ref(KCode));
}

void dataAnalysisPipeline::startAnalysis(int cycle) {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return;
    }
    impl->startAnalysis(cycle);
}

int dataAnalysisPipeline::checkAnalysis() {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return 0;
    }
    return impl->checkAnalysis();
}

int dataAnalysisPipeline::waitForAnalysis() {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return 0;
    }
    return impl->waitForAnalysis();
}

dataAnalysisPipeline::~dataAnalysisPipeline() {
    
}
    
} // namespace dataAnalysis







