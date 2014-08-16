---
layout: page
title: 
permalink: /julia_cheetsheet/
---

## Julia Support: Cheetsheet

### Constant: Access Methods

  - DimmWitted.AC_ROW
  - DimmWitted.AC_COL
  - DimmWitted.AC_C2R

### Constant: Model Replication Strategy

  - DimmWitted.MR_PERCORE
  - DimmWitted.MR_PERNODE
  - DimmWitted.MR_PERMACHINE
  - DimmWitted.MR_SINGLETHREAD_DEBUG

### Constant: Data Replication Strategy

  - DimmWitted.DR_FULL
  - DimmWitted.DR_SHARDING

### Function: `DimmWitted.open()` -> DimmWitted object

  - Dense: `open(Array{DATATYPE,2}, Array{MODELTYPE,1}, MR, DR, AC [, Array{SHAREDATA_TYPE,1}])`
  - Sparse: `open(SparseMatrixCSC{DATATYPE,Int64}, Array{MODELTYPE,1}, MR, DR, AC [, Array{SHAREDATA_TYPE,1}]`

### Function: `DimmWitted.register_*` -> Function handle

  - DimmWitted.register_row(DimmWitted object, function)
  - DimmWitted.register_col(DimmWitted object, function)
  - DimmWitted.register_c2r(DimmWitted object, function)
  - DimmWitted.register_model_avg(Function handle, function)

See [this page](/dw/julia/) and [this page](/dw/julia_scd/) for schema of `function`.

See [this page](/dw/julia_global/) for the meaning of the following three functions.

  - DimmWitted.register_row2
  - DimmWitted.register_col2
  - DimmWitted.register_c2r2


### Function: `DimmWitted.exec()` -> Cdouble as the result

  - `exec(DimmWitted object, function handle)`

