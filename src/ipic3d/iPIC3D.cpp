/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "MPIdata.h"
#include "iPic3D.h"
#include "debug.h"
#include "TimeTasks.h"
#include <stdio.h>

#include "dataAnalysis.cuh"

using namespace iPic3D;

int main(int argc, char **argv) {

 MPIdata::init(&argc, &argv);
 {

  iPic3D::c_Solver KCode;
  KCode.Init(argc, argv); //! load param from file, init the grid, fields
  dataAnalysis::dataAnalysisPipeline DA(KCode); // has to be created after KCode.Init()


  timeTasks.resetCycle(); //reset timer
  KCode.CalculateMoments();
  for (int i = KCode.FirstCycle(); i < KCode.LastCycle(); i++) {

    if (KCode.get_myrank() == 0)
      printf(" ======= Cycle %d ======= \n",i);

    timeTasks.resetCycle();

    DA.startAnalysis(i);
    KCode.CalculateField(i); // E field
    DA.waitForAnalysis();

    KCode.ParticlesMoverMomentAsync(); // launch Mover and Moment kernels
    KCode.CalculateB(); // B field
    KCode.CalculateMomentsAwait(); // wait for Mover and Moment kernels
    
    KCode.WriteOutput(i);
    
#ifdef LOG_TASKS_TOTAL_TIME
    timeTasks.print_cycle_times(i); // print out total time for all tasks
#endif
  }

#ifdef LOG_TASKS_TOTAL_TIME
    timeTasks.print_tasks_total_times();
#endif

  KCode.Finalize();
 }
 // close MPI
 MPIdata::instance().finalize_mpi();

 return 0;
}
