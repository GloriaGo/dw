/**
Copyright 2014 Hazy Research (http://i.stanford.edu/hazy)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
**/


#ifndef _SCHEDULER_PERCORE_H
#define _SCHEDULER_PERCORE_H

#include "engine/scheduler.h"

template<class RDTYPE, class WRTYPE>
double _percore_run_map(double (*p_map) (long, const RDTYPE * const, WRTYPE * const),
  const RDTYPE * const RDPTR, 
              WRTYPE * const WRPTR, 
              const long * const tasks,
              long start, long end){
  double rs = 0.0;
  for(long i=start;i<end;i++){
    rs += p_map(tasks[i], RDPTR, WRPTR);
  }
  return rs;
}

template<class RDTYPE,
         class WRTYPE,
         DataReplType DATAREPL>
class DWRun<RDTYPE, WRTYPE, DW_PERCORE,
        DATAREPL> {  
public:
  
  const RDTYPE * const RDPTR;

  WRTYPE * const WRPTR;

  WRTYPE ** model_replicas;

  void (*p_model_allocator) (WRTYPE ** const, const WRTYPE * const);

  DWRun(const RDTYPE * const _RDPTR, WRTYPE * const _WRPTR,
        void (*_p_model_allocator) (WRTYPE ** const, const WRTYPE * const)
    ):
    RDPTR(_RDPTR), WRPTR(_WRPTR),
    p_model_allocator(_p_model_allocator)
  {}


  void prepare(){
    //int n_numa_nodes = numa_max_node();
    long n_sharding = getNumberOfCores();
    model_replicas = new WRTYPE*[n_sharding+1];
    for(int i=0;i<n_sharding;i++){
      p_model_allocator(&model_replicas[i], WRPTR);
    }
    model_replicas[n_sharding] = WRPTR;
    //std::cout << "~~~~~~" << std::endl;
  }

  double exec(const long * const tasks, int ntasks,
    double (*p_map) (long, const RDTYPE * const, WRTYPE * const),
         void (*p_comm) (WRTYPE ** const, int, int),
         void (*p_finalize) (WRTYPE * const, WRTYPE ** const, int)
    ){

    std::vector<std::future<double>> futures;

    long n_sharding = getNumberOfCores();

    double rs = 0.0;
    
    for(int i_sharding=0;i_sharding<n_sharding;i_sharding++){
      long start = ((long)(ntasks/n_sharding)+1) * i_sharding;
      long end = ((long)(ntasks/n_sharding)+1) * (i_sharding+1);
      end = end >= ntasks ? ntasks : end;
      //threads.push_back(std::thread(_percore_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, model_replicas[i_sharding], tasks, start, end));
      if(DATAREPL == DW_FULL){
        futures.push_back(std::async(_percore_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, model_replicas[i_sharding], tasks, 0, ntasks));
      }else{
        futures.push_back(std::async(_percore_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, model_replicas[i_sharding], tasks, start, end));
      }
    }

    for(int i=0;i<n_sharding;i++){
      rs += futures[i].get();
    }

    //p_finalize(WRPTR, model_replicas, n_sharding);
    p_comm(model_replicas, n_sharding, n_sharding);

    return rs;
  }

};


#endif





