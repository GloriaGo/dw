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

#ifndef _CNN_SPARSE_H
#define _CNN_SPARSE_H

#include "dimmwitted.h"
#include "neural_obj.h"
#include <random>

double learn_rate=0.1;
double reg_rate=0.000;
double tanh_bias=0.001;

class cnn_layer_model{
public:
  neural_network * network;
  long size; 
  double * current_grads;
  double * values;
  weight * weights;
  cnn_layer_model * next;
  cnn_layer_model * prev;
  long layer;
  long num_input;
};

double back_gradient(const SparseVector<double>* const ex, cnn_layer_model* const p_model){
  // cout << "Calculating back gradient\n";

  long hedge_ind=ex->p[0];
  hyper_edge* hedge=&p_model->network->hedges[hedge_ind];
  if(hedge->factor_function == 1011){ // Logistic Loss

    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=p_model->network->variables[out_id].start_ind;
    long offset=out_x*p_model->network->variables[out_id].num_cols+out_y;


    double init_value=p_model->network->variables[out_id].init_value[offset];
    double current_value=p_model->values[start_ind+offset];

    // Following is least squares
    double grad =2*(current_value-init_value);
    // show(grad);

    // double exp_one_minus_2y = 
    //   exp((1.0 - 2*init_value)*current_value);

    std::cout << "INIT:" << init_value << "   CURRENT:" << current_value
      << "   grad:" << grad << std::endl;

    p_model->current_grads[start_ind+offset] = grad;
    // p_model->current_grads[start_ind+offset] = exp_one_minus_2y/(1.0+exp_one_minus_2y)*
    //                             (1.0 - 2*init_value);
    // show(start_ind);
    // show(offset);
    // show(p_model->current_grads[start_ind+offset]);
  }else if(hedge->factor_function == 1010){ // Softmax Loss
    double sum_y=0;
    for(int i=0; i<ex->n-2; i++){
      long in_mat_id = hedge->in_mat_ids[i];
      long in_x=hedge->in_center_xs[i];
      long in_y=hedge->in_center_ys[i];
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long offset=in_x*p_model->network->variables[in_mat_id].num_cols+in_y;
      double current_value=p_model->values[start_ind+offset];
      sum_y+=current_value;
    }

    for(int i=0; i<ex->n-2; i++){
      const weight * weight_i = &p_model->weights[hedge->weight_ids[i]];
      long in_mat_id = hedge->in_mat_ids[i];
      long in_x=hedge->in_center_xs[i];
      long in_y=hedge->in_center_ys[i];
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long offset=in_x*p_model->network->variables[in_mat_id].num_cols+in_y;
      double init_value=p_model->network->variables[in_mat_id].init_value[offset];
      double current_value=p_model->values[start_ind+offset];
      if(init_value==i){
        // show(-(sum_y-current_value)/(current_value*sum_y));
        p_model->current_grads[start_ind+offset]=-(sum_y-current_value)/(current_value*sum_y);
      }
      else{
        // show(1.0/sum_y);
        p_model->current_grads[start_ind+offset]=1.0/sum_y;
      }
    }
  }else if(hedge->factor_function == 1000){ // Conv
    double sum = 0.0;
    for(int i=0; i<ex->n-2; i++){
      const weight * weight_i = &p_model->weights[hedge->weight_ids[i]];
      long in_mat_id = hedge->in_mat_ids[i];
      long in_x=hedge->in_center_xs[i];
      long in_y=hedge->in_center_ys[i];
      long start_ind=p_model->network->variables[in_mat_id].start_ind;

      for(int r=0; r<weight_i->num_rows; r++)
        for(int c=0; c<weight_i->num_cols; c++){
          long offset=(in_x+r)*p_model->network->variables[in_mat_id].num_cols+(in_y+c);
          sum += weight_i->values[r][c] * p_model->values[start_ind+offset];
          // std::cout << "~~~~" << weight_i->values[r][c] << "     *      " << p_model->values[start_ind+offset] << std::endl;
        }
       sum += weight_i->bias;
    }
    // show(sum);

    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    // show(out_id);
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;

    double grad1 = (1.0/(1.0+exp(-sum))) *  (1.0-(1.0/(1.0+exp(-sum)))); //SIGMOID 
    // double grad1 = 1- ((exp(sum)-exp(-sum))/(exp(sum) + exp(-sum))) * ((exp(sum)-exp(-sum))/(exp(sum) + exp(-sum))); //TANH
    grad1/=p_model->num_input;
    // double grad1 = 2*sum;
    // show(grad1);
    grad1 = grad1 * n_model->current_grads[start_ind+offset];
    // show(start_ind+offset);
    // show(n_model->current_grads[start_ind+offset]);
    // show(grad1);
    //       show(p_model);
    //       show(n_model);


    // for(int i=0; i<p_model->size; i++)
    //   p_model->current_grads[i]=0;

    for(int i=0; i<ex->n-2; i++){
      weight * weight_i = &p_model->weights[hedge->weight_ids[i]];
      long in_mat_id = hedge->in_mat_ids[i];
      long in_x=hedge->in_center_xs[i];
      long in_y=hedge->in_center_ys[i];
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      for(int r=0; r<weight_i->num_rows; r++)
        for(int c=0; c<weight_i->num_cols; c++){
          long offset=(in_x+r)*p_model->network->variables[in_mat_id].num_cols+(in_y+c);
          p_model->current_grads[start_ind+offset] += grad1 * weight_i->values[r][c];
          weight_i->values[r][c]-=learn_rate * grad1*p_model->values[start_ind+offset];//+reg_rate*weight_i->values[r][c];
          // if(hedge->layer==0)
            // std::cout << "#####" << p_model->values[start_ind+offset] << std::endl;
            // std::cout << "    ~~  " << "GRAD: " << p_model->current_grads[start_ind+offset] << " values : " << p_model->values[start_ind+offset] 
            //           << " new_weight_delta: " <<  0.01 * grad1*p_model->values[start_ind+offset]
            //           << " start_ind+offset: " << start_ind+offset
            //           << " grad_w: " << grad1*p_model->values[start_ind+offset]<< std::endl;
        }
      weight_i->bias=weight_i->bias-learn_rate * grad1; //TODO Check

      
      //std::cout << "    #   " << grad1 << "    " << current_values[neuron_id] << std::endl;
      //std::cout << connection.weight_ids[i] << "--->" << weight.value << std::endl;
    }

  }else if(hedge->factor_function == 1020){ // softmax
    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;

    double grad1 = n_model->values[start_ind+offset];
    cout.precision(15);
    cout << " value last layer: " << grad1 << "   ";
    grad1 = grad1 * n_model->current_grads[start_ind+offset];
    cout << "value*grad last layer: " << grad1 << endl;

    for(int i=0; i<ex->n-2; i++){
      weight * weight_i = &p_model->weights[hedge->weight_ids[i]];
      long in_mat_id = hedge->in_mat_ids[i];
      long in_x=hedge->in_center_xs[i];
      long in_y=hedge->in_center_ys[i];
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      for(int r=0; r<weight_i->num_rows; r++)
        for(int c=0; c<weight_i->num_cols; c++){
          long offset=(in_x+r)*p_model->network->variables[in_mat_id].num_cols+(in_y+c);
          p_model->current_grads[start_ind+offset] += grad1 * weight_i->values[r][c];
          // show(grad1 * weight_i->values[r][c]);
          // show(grad1 * weight_i->values[r][c]);
          weight_i->values[r][c]-=learn_rate * grad1*p_model->values[start_ind+offset];//+reg_rate*weight_i->values[r][c];
          cout << "grad* value of softmax layer: " << grad1*p_model->values[start_ind+offset] << endl;
         }
       weight_i->bias = weight_i->bias - learn_rate * grad1; //TODOOOOOO check
    }

  }else{
    std::cout << "FUNCTION ID " << hedge->factor_function << " NOT DEFINED!" << std::endl;
    assert(false);
  }
  return 0.0;
}

double forward_propogate(const SparseVector<double>* const ex, cnn_layer_model* const p_model){
  // cout << "Forward propogating\n";

  //std::cout << connection.func_id << std::endl;
  long hedge_ind=ex->p[0];
  hyper_edge* hedge=&p_model->network->hedges[hedge_ind];
  if(hedge->factor_function == 1000){ // Conv
    // std::cout << ex->n-2 << " == " << hedge->in_mat_ids.size() << std::endl;
    double sum = 0.0;

    for(int i=0; i<ex->n-2; i++){
      const weight * weight_i = &p_model->weights[hedge->weight_ids[i]];
      long in_mat_id = hedge->in_mat_ids[i];
      long in_x=hedge->in_center_xs[i];
      long in_y=hedge->in_center_ys[i];
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      for(int r=0; r<weight_i->num_rows; r++)
        for(int c=0; c<weight_i->num_cols; c++){
          long offset=(in_x+r)*p_model->network->variables[in_mat_id].num_cols+(in_y+c);
          sum += weight_i->values[r][c] * p_model->values[start_ind+offset];
          // if(hedge->layer==1){
            // show(in_mat_id);
            // show(p_model->values[start_ind+offset]);
          // }
        }
      sum += weight_i->bias;
      // show(sum);
    }

    // if(hedge->layer==5){
    //   show(hedge->in_mat_ids[0]);
    //   show(ex->p[1]);
    //   show(hedge->in_center_xs[0]);
    //   show(hedge->in_center_ys[0]);
    //   std::cout << "F " << sum << " --> " << (1.0 - exp(-2*sum))/(1.0 + exp(-2*sum)) << std::endl;
    // }
    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
    n_model->values[start_ind+offset] = (1.0)/(1.0 + exp(-sum)); //Sigmoid
    // n_model->values[start_ind+offset] = (exp(sum)-exp(-sum))/(exp(sum) + exp(-sum)); //Tanh
    // n_model->values[start_ind+offset] = (exp(sum)-exp(-sum))/(exp(sum) + exp(-sum)) + tanh_bias*sum; //Tanh


    // n_model->values[start_ind+offset] = sum*sum;


   // if(hedge->layer==5){
   //  cout << out_id << endl;
   //  for(long i=0 ;i <n_model->network->variables[out_id].num_rows; i++)
   //    for(long j=0; j<n_model->network->variables[out_id].num_cols; j++){
   //      long start_ind=n_model->network->variables[out_id].start_ind;
   //      long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
   //          cout << p_model->values[start_ind+offset] <<  " ";
   //        }
   //    cout << endl;
   //  }

    // n_model->values[start_ind+offset] = (1.0 - exp(-2*sum))/(1.0 + exp(-2*sum));
    // if(hedge->layer==0){
    //   show(out_id);
    //   show(sum);
    //   show(n_model->values[start_ind+offset]);
    // }

    // if(hedge->layer==5){
    //   show(out_id);
    //   show(sum);
    //   show(n_model->values[start_ind+offset]);
    // }

    // if(hedge->layer==4 && out_x==10 && out_y==10){
    //   show(out_id);
    //   show(sum);
    //   show(n_model->values[start_ind+offset]);
    // }

  }else if(hedge->factor_function == 1020){ // softmax
    double sum = 0.0;

    for(int i=0; i<ex->n-2; i++){
      const weight * weight_i = &p_model->weights[hedge->weight_ids[i]];
      long in_mat_id = hedge->in_mat_ids[i];
      long in_x=hedge->in_center_xs[i];
      long in_y=hedge->in_center_ys[i];
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      for(int r=0; r<weight_i->num_rows; r++)
        for(int c=0; c<weight_i->num_cols; c++){
          long offset=(in_x+r)*p_model->network->variables[in_mat_id].num_cols+(in_y+c);
          sum += weight_i->values[r][c] * p_model->values[start_ind+offset];
        }
      sum += weight_i->bias;
    }
    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
    n_model->values[start_ind+offset] = exp(sum);
    // cout << "SUM: " << exp(sum) << endl;
  }else{
    std::cout << "FUNCTION ID " << hedge->factor_function << " NOT DEFINED!" << std::endl;
    assert(false);
  }
  return 0.0;
}


double error(const SparseVector<double>* const ex, cnn_layer_model* const p_model){
  // cout << "Calculating error ..." << endl;

  long hedge_ind=ex->p[0];
  hyper_edge* hedge=&(p_model->network->hedges[hedge_ind]);

  if(hedge->factor_function == 1011){ // Logistic Regression // Least Squares
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=p_model->network->variables[out_id].start_ind;
    long offset=out_x*p_model->network->variables[out_id].num_cols+out_y;

    double init_value=p_model->network->variables[out_id].init_value[offset];
    double current_value=p_model->values[start_ind+offset];

    std::cout << "E  " << current_value << "   " <<  init_value << std::endl;
    std::cout << "Error  " << (current_value - init_value) * (current_value - init_value) << std::endl;

    // Least Squares loss
    return (current_value - init_value) * (current_value - init_value);

    // return log(1.0 + exp((1-2*init_value)*current_value));

  }else if(hedge->factor_function == 1010){ // softmax
    double denom=0;
    double numer=0;
    double init_value;
    for(int i=0; i<ex->n-2; i++){
      const weight * weight_i = &p_model->weights[hedge->weight_ids[i]];
      long in_mat_id = hedge->in_mat_ids[i];
      long in_x=hedge->in_center_xs[i];
      long in_y=hedge->in_center_ys[i];
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long offset=in_x*p_model->network->variables[in_mat_id].num_cols+in_y;
      init_value=p_model->network->variables[in_mat_id].init_value[offset];
      double current_value=p_model->values[start_ind+offset];
      if(init_value==i)
        numer=current_value;
      denom+=current_value;
    }
    double l2_norm=0;
    for(int i=0; i<p_model->network->num_weights; i++){
      weight * weight_i = &p_model->weights[i];
      for(int r=0; r<weight_i->num_rows; r++)
        for(int c=0; c<weight_i->num_cols; c++){
          l2_norm += (weight_i->values[r][c] * weight_i->values[r][c]);
        }
    }
    double prob=numer/denom;
    std::cout.precision(15);
    std::cout << "Prob  " << prob << " for class: " << init_value << std::endl;

    // double loss=-log(numer/denom)+l2_norm*reg_rate/2.0; //REG
    double loss=-log(prob);
    // std::cout << "Error SM:  " << loss << std::endl;

    loss/=p_model->num_input;

    return loss;
  }else{
    std::cout << "FUNCTION ID " << hedge->factor_function << " NOT DEFINED!" << std::endl;
    assert(false);
  }
  return 0.0;
}

double v_error(const SparseVector<double>* const ex, cnn_layer_model* const p_model){
  // cout << "Calculating v error ..." << endl;

  long hedge_ind=ex->p[0];
  hyper_edge* hedge=&(p_model->network->hedges[hedge_ind]);

  if(hedge->factor_function == 1011){ // Logistic Regression // Least Squares
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=p_model->network->variables[out_id].start_ind;
    long offset=out_x*p_model->network->variables[out_id].num_cols+out_y;

    double init_value=p_model->network->variables[out_id].init_value[offset];
    double current_value=p_model->values[start_ind+offset];

    std::cout << "E  " << current_value << "   " <<  init_value << std::endl;
    std::cout << "Error  " << (current_value - init_value) * (current_value - init_value) << std::endl;

    // Least Squares loss
    return (current_value - init_value) * (current_value - init_value);

    // return log(1.0 + exp((1-2*init_value)*current_value));

  }else if(hedge->factor_function == 1010){ // softmax
    double denom=0;
    double numer=0;
    double init_value;
    for(int i=0; i<ex->n-2; i++){
      const weight * weight_i = &p_model->weights[hedge->weight_ids[i]];
      long in_mat_id = hedge->in_mat_ids[i];
      long in_x=hedge->in_center_xs[i];
      long in_y=hedge->in_center_ys[i];
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long offset=in_x*p_model->network->variables[in_mat_id].num_cols+in_y;
      init_value=p_model->network->variables[in_mat_id].init_value[offset];
      double current_value=p_model->values[start_ind+offset];
      if(init_value==i)
        numer=current_value;
      denom+=current_value;
    }
    double l2_norm=0;
    for(int i=0; i<p_model->network->num_weights; i++){
      weight * weight_i = &p_model->weights[i];
      for(int r=0; r<weight_i->num_rows; r++)
        for(int c=0; c<weight_i->num_cols; c++){
          l2_norm += (weight_i->values[r][c] * weight_i->values[r][c]);
        }
    }

    double prob=numer/denom;
    // std::cout << "Prob  " << prob << " for class: " << init_value << std::endl;

    if(prob>0.5)
      return 100.0/p_model->num_input;
    else return 0;
    // return loss;
  }else{
    std::cout << "FUNCTION ID " << hedge->factor_function << " NOT DEFINED!" << std::endl;
    assert(false);
  }
  return 0.0;
}


template<ModelReplType MODELREPL, DataReplType DATAREPL>
double cnn_sparse(neural_network &network){
  int num_layer=7;
  cout.precision(15);
  
  long * nexp=new long[num_layer];
  long * nfeat=new long[num_layer];
  double ** examples = new double * [num_layer];
  long ** cols = new long * [num_layer];
  long ** rows = new long * [num_layer];
  cnn_layer_model * models= new cnn_layer_model [num_layer];
  unsigned int * f_handle_f_prop=new unsigned int [num_layer];
  unsigned int * f_handle_b_prop=new unsigned int [num_layer];
  unsigned int f_handle_error;
  unsigned int f_handle_v_error;


  SparseDimmWitted<double, cnn_layer_model, MODELREPL, DATAREPL, DW_ACCESS_ROW> ** dw_l=
    new SparseDimmWitted<double, cnn_layer_model, MODELREPL, DATAREPL, DW_ACCESS_ROW> *[num_layer];


  for(int l=0; l<num_layer; l++){
    //calculate number of variables in layer i(nexp)
    nfeat[l]=0;
    for(int i=0; i<network.num_vars; i++){
      // show(network.variables[i].layer);
      if(network.variables[i].layer==l){
        nfeat[l]++;
      }
    }

    //calculate number of hedges with input variable in layer i (nexp)
    nexp[l]=0;
    for(int i=0; i<network.num_hedges; i++){
      for(int j=0; j<network.hedges[i].in_mat_ids.size(); j++){
        if(network.variables[network.hedges[i].in_mat_ids[j]].layer==l){
          nexp[l]++;
          network.hedges[i].layer=l; //initialize the layer of each hedge
          break;
        }
      }
    }
    examples[l] = new double[nexp[l]*(nfeat[l]+2)];
    cols[l] = new long[nexp[l]*(nfeat[l]+2)];
    rows[l] = new long[nexp[l]];

  }
  

  int * num_var=new int[num_layer];
  for(int l=0; l<num_layer; l++){

    cout << "**************** LAYER : " << l << " ****************\n";
    num_var[l]=0;

    //Initializing variable of layer l starting index for the model (Used in the gradient and value arrays)
    long var_layer_ind_ct=0;
    for(int i=0; i<network.num_vars; i++)
      if(network.variables[i].layer==l){
        // show(i);
        network.variables[i].start_ind=num_var[l];
        // show(network.variables[i].start_ind);
        network.variables[i].var_layer_ind=var_layer_ind_ct++;
        num_var[l]+=network.variables[i].num_rows*network.variables[i].num_cols;
      }
    // show(num_var[l]);
    // show(var_layer_ind_ct);

    cout << "Initialize data matrix ..." << endl;
    // Initialize Examples/Cols/Rows arrays of layer l
    long col_i=0,row_i=0;
    for(int i=0; i<network.num_hedges; i++){
      if(network.hedges[i].layer==l){
        rows[l][row_i++]=col_i;
        network.hedges[i].start_ind=col_i;

        //Examples=[hedge_id,in_id1,in_id2,...,in_idn,out_id]
        examples[l][col_i]=i; //hedge id
        cols[l][col_i]=0;
        col_i++;

        for(int j=0; j<network.hedges[i].in_mat_ids.size(); j++){
          examples[l][col_i]=network.hedges[i].in_mat_ids[j]; //Input -> 1
          long ind=network.hedges[i].in_mat_ids[j];
          // show(ind);
          cols[l][col_i]=network.variables[ind].var_layer_ind+1;
          col_i++;
        }

        examples[l][col_i]=network.hedges[i].out_mat_id; //Output -> 2
        cols[l][col_i]=nfeat[l]+1;
        col_i++;
      }
    }



    cout << "Initializing workspace ...\n";
    

    //Initilize Model for layer l
    models[l].network=&network;
    models[l].current_grads=new double[num_var[l]];
    //Init values
    models[l].values=new double[num_var[l]];
    for(long i=0; i<network.num_vars; i++)
      if(network.variables[i].layer==l){
        long start_ind=network.variables[i].start_ind;
        for(int r=0; r<network.variables[i].num_rows; r++)
          for(int c=0; c<network.variables[i].num_cols; c++){

            long offset=r*network.variables[i].num_cols+c;
            models[l].values[start_ind+offset]=network.variables[i].init_value[offset];
            // if(i==74 || i==75){
            //   cout << models[l].values[start_ind+offset] << " ";
            // }
          }
        // if(i==74 || i==75){
        //   show(start_ind);
        // }
      }
    models[l].size=num_var[l];
    models[l].num_input=nfeat[0];
    uniform_real_distribution<double> distribution1(0.1,0.1);
    uniform_real_distribution<double> distribution5(0.1,0.1);
    default_random_engine generator;
    //Init weights
    models[l].weights=new weight[network.num_weights];
    for(long i=0; i<network.num_weights; i++){
      models[l].weights[i].weight_id=i;
      models[l].weights[i].is_fixed=network.weights[i].is_fixed;
      models[l].weights[i].num_rows=network.weights[i].num_rows;
      models[l].weights[i].num_cols=network.weights[i].num_cols;
      models[l].weights[i].bias=0;
      models[l].weights[i].values=new double * [network.weights[i].num_rows];
      // show(network.weights[i].num_rows);
      for(int j=0; j<models[l].weights[i].num_rows; j++){
        models[l].weights[i].values[j]=new double [network.weights[i].num_cols];
        for(int k=0; k<network.weights[i].num_cols; k++){
          // models[l].weights[i].values[j][k]=network.weights[i].initial_value;
          // uniform_real_distribution<double> distribution(-1,1);

          if(l<5){


            double number = distribution1(generator);
            do
              number = distribution1(generator);
            while ((number<-5)&&(number>5));
            models[l].weights[i].values[j][k]=number;
          }
          else{
           

            double number = distribution5(generator);
            do
              number = distribution5(generator);
            while ((number<-5)&&(number>5));
            models[l].weights[i].values[j][k]=number;
          }
          // models[l].weights[i].values[j][k]=0.1;
          if(models[l].weights[i].is_fixed==1){
              models[l].weights[i].values[j][k]=network.weights[i].initial_value;
          }
        }
      }
    }
    //Init next/prev pointers to the next/prev layer models
    if(l!=0)
      models[l].prev=&models[l-1];
    if(l!=num_layer-1)
      models[l].next=&models[l+1];
    models[l].layer=l;


    cout << "Initializing DW ...\n";

    //Initialize DW for layer l

      //_n_rows, long _n_cols, long _n_elems
      //nexp[l], nfeat[l], col_i

    // show(nexp[l]);
    // for(int i=0; i<nexp[l]; i++){
    //   cout << rows[l][i] << " ";
    // }
    //     cout << endl;

    // show(nfeat[l]);
    // for(int i=0; i<col_i; i++){
    //   cout << cols[l][i] << " ";
    // }
    //     cout << endl;

    // show(col_i);
    // for(int i=0; i<col_i; i++){
    //   cout << examples[l][i] << " ";
    // }
    // cout << endl;

    // show(l);
    // show(nexp[l]);
    // show(nfeat[l]);

    dw_l[l]=new SparseDimmWitted<double, cnn_layer_model, MODELREPL, DATAREPL, DW_ACCESS_ROW> 
      (examples[l], rows[l], cols[l], nexp[l], nfeat[l], col_i, &models[l]);

    cout << "Registering functions ...\n";
    //Register the functions
    if(l!=num_layer-1)
      f_handle_f_prop[l] = dw_l[l]->register_row(forward_propogate);
    f_handle_b_prop[l] = dw_l[l]->register_row(back_gradient);
  }
  f_handle_error = dw_l[num_layer-1]->register_row(error);
  f_handle_v_error=dw_l[num_layer-1]->register_row(v_error);
  double loss=0;
  double last_loss=-1;
  double validation_error=0;



  for(int i_epoch=0;i_epoch<2;i_epoch++){
    cout << "EPOCH num : " << i_epoch << endl;


    for(int l=0; l<num_layer-1; l++){
      cout << "Forward propogate layer : " << l << endl;
      dw_l[l]->exec(f_handle_f_prop[l]);
      cout<< "Values:\n";
      int size;
      for(int j=0; j<models[l].network->num_vars; j++)
        if(models[l].network->variables[j].layer==l)
          size=models[l].network->variables[j].num_rows*models[l].network->variables[j].num_cols;
      show(size);
      for(int i=0; i<models[l].size/size; i++){
        if(i==0 || i==models[l].size/size-1)
          cout << "[";
        for(int j=0; j<size; j++)
          if(i==0 || i==models[l].size/size-1)
            cout << models[l].values[i*size+j] << ", ";
        if(i==0 || i==models[l].size/size-1)
          cout << "]" << endl;
      }
      // cout << "Grads:\n";
      // for(int i=0; i<models[l].size/size; i++){
      //   // if(i==0 || i==models[l].size/size-1)
      //     cout << "[";
      //   for(int j=0; j<size; j++)
      //     // if(i==0 || i==models[l].size/size-1)
      //       cout << models[l].current_grads[i*size+j] << ", ";
      //   // if(i==0 || i==models[l].size/size-1)
      //     cout << "]" << endl;
      // }
    }
    // cout<< "Values: \n";
    // for(int i=0; i<models[num_layer-1].size; i++)
    //   cout << "[" << models[num_layer-1].values[i] << "]\n";
    // cout << endl;
    // cout<< "Grads: \n";
    // for(int i=0; i<models[num_layer-1].size; i++)
    //   cout << "[" << models[num_layer-1].current_grads[i] << "]\n";
    // cout << endl;

    loss=dw_l[num_layer-1]->exec(f_handle_error);
    validation_error=dw_l[num_layer-1]->exec(f_handle_v_error);

    // loss=dw_l[num_layer-1]->exec(f_handle_error);
    show(loss);
    show(validation_error);

      

    for(int l=num_layer-1; l>=0; l--){
      cout << "************ Backward propogate layer : " << l << "************" << endl;
      for(int i=0; i<models[l].size; i++)
        models[l].current_grads[i]=0;
      dw_l[l]->exec(f_handle_b_prop[l]);
    }




    // for(int l=0; l<num_layer-1; l++){
    //   std::cout.precision(15);
    //   cout << "Layer " << l << " weights:" << endl;
    //   for(int i=0; i<models[l].network->num_hedges; i++)
    //     if(models[l].network->hedges[i].layer==l){
    //       const weight * weight_i = &models[l].weights[models[l].network->hedges[i].weight_ids[0]];
    //       for(int j=0; j<weight_i->num_rows; j++){
    //         for(int k=0; k<weight_i->num_cols; k++){
    //           std::cout.precision(15);
    //           cout << weight_i->values[j][k] << " ";
    //         }
    //        cout<< endl;
    //       }
    //       std::cout.precision(15);
    //       cout << "bias: " << weight_i->bias << endl;
    //       break;
    //     }
    //   cout << endl;
    // }
    // for(int i=0; i<models[num_layer-2].network->num_hedges; i++)
    //   if(models[num_layer-2].network->hedges[i].layer==num_layer-2){
    //     for(int l=0; l<models[num_layer-2].network->hedges[i].num_inputs; l++){
    //       const weight * weight_i = &models[num_layer-2].weights[models[num_layer-2].network->hedges[i].weight_ids[l]];
    //       for(int j=0; j<weight_i->num_rows; j++){
    //         for(int k=0; k<weight_i->num_cols; k++){
    //           std::cout.precision(15);
    //           cout << weight_i->values[j][k] << " ";
    //         }
    //        cout<< endl;
    //       }
    //       std::cout.precision(15);
    //       cout << "bias: " << weight_i->bias << endl;
    //     }
    //     break;
    //   }
    // if((loss-last_loss)*(loss-last_loss)<1E-14)
    //   break;
    last_loss=loss;

  }
  return 0.0;
}

#endif