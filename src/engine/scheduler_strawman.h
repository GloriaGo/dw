// Copyright 2014 Hazy Research (http://i.stanford.edu/hazy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef _SCHEDULER_STRAWMAN_H
#define _SCHEDULER_STRAWMAN_H

#include "engine/scheduler.h"

/**
 * \brief A specialization of DWRun that is a single
 * thread version of what DWRun did, merely for the
 * purpose of debugging.
 */
template<class RDTYPE,
         class WRTYPE,
         DataReplType DATAREPL>
class DWRun<RDTYPE, WRTYPE, DW_MODELREPL_SINGLETHREAD_DEBUG, DATAREPL> {  
public:
  
  const RDTYPE * const RDPTR;

  WRTYPE * const WRPTR;

  void (*p_model_allocator) (WRTYPE ** const, const WRTYPE * const);

  DWRun(const RDTYPE * const _RDPTR, WRTYPE * const _WRPTR,
        void (*_p_model_allocator) (WRTYPE ** const, const WRTYPE * const)
    ):
    RDPTR(_RDPTR), WRPTR(_WRPTR),
    p_model_allocator(_p_model_allocator)
  {}

  void prepare(){

  }

  double exec(const long * const tasks, int ntasks,
             double (*p_map) (long, const RDTYPE * const, WRTYPE * const),
         void (*p_comm) (WRTYPE ** const, int, int),
         void (*p_finalize) (WRTYPE * const, WRTYPE ** const, int)
    ){
    double rs = 0.0;
    for(long i=0;i<ntasks;i++){
      rs += p_map(tasks[i], RDPTR, WRPTR);
    }
    return rs;
  }

};


#endif