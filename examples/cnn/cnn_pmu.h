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

#define HAZY_MSR_BUILD
// note: you'll have to chance these paths to point to the right place
#include "msraccess-inl.h"
#include "uncorepmu-inl.h"


#include "dimmwitted.h"
#include "neural_obj.h"
#include <chrono>
#include <ctime>
#include <random>

///////////////////////////////// Performance Monitor ///////////////////////////////////
long num_flop_add=0;
long num_flop_mul=0;
long num_flop_div=0;
long num_flop_com=0;
#define inc_add(x)  num_flop_add+=x
#define inc_mul(x)  num_flop_mul+=x
#define inc_div(x)  num_flop_div+=x
#define inc_com(x)  num_flop_com+=x
#define reset_flop() num_flop_add=num_flop_mul=num_flop_div=num_flop_com=0
#define print_flop(x) cout << "Number of sum, mul, div, complex FLOP per image: " << num_flop_add/x << ", " << num_flop_mul/x << ", " << num_flop_div/x << ", " << num_flop_com/x << endl;
#define print_flops(x,t) cout << "FLOPS (sum, mul, div, complex, total): " << num_flop_add/x/t << ", " << num_flop_mul/x/t << ", " << num_flop_div/x/t << ", " << num_flop_com/x/t << ", " << num_flop_add/x/t+num_flop_mul/x/t+num_flop_div/x/t+num_flop_com/x/t << endl;

///////////////////////////////// Throughput Monitor ///////////////////////////////////
long mem_read_bytes=0;
long mem_write_bytes=0;
#define inc_read(x)  mem_read_bytes+=x
#define inc_write(x)  mem_write_bytes+=x
#define reset_mem() mem_read_bytes=mem_write_bytes=0
#define print_mem(x) cout << "Bytes read, write of memory per image: " << mem_read_bytes/x << ", " << mem_write_bytes/x << endl;
#define print_mems(x,t) cout << "Throughput (read, write, total): " << mem_read_bytes/x/t << ", " << mem_write_bytes/x/t << ", " << mem_read_bytes/x/t+mem_write_bytes/x/t << endl;






double learn_rate=0.1;
double reg_rate=0.000;
double tanh_bias=0.001;
const double EPS=1E-5;

class cnn_layer_model{
public:
  neural_network * network;
  long size;  //number of variables in layer l;
  double * current_grads;
  double * values;
  weight * weights;   //All the weights are stored for each layer!
  weight * new_weights;
  cnn_layer_model * next;
  cnn_layer_model * prev;
  long layer;
  long num_input;   //Size of current_grads/values/ ... 

  long mem_size(){
    long wei_size=0;
    for(int i=0; i<network->num_weights; i++)
      wei_size+=weights[i].mem_size();
    return sizeof(neural_network *)+ sizeof(long)*3+ sizeof(double *)*2+
            sizeof(weight *)*2+ sizeof(cnn_layer_model *)*2;
  }

};

double back_gradient(const SparseVector<double> * const ex, cnn_layer_model* const p_model){
  // cout << "Calculating back gradient\n";
  long hedge_ind=ex->p[0];
  hyper_edge* hedge=&p_model->network->hedges[hedge_ind];

  inc_read(sizeof(ex->p)); // reading p[0]
  inc_read(sizeof(hyper_edge)); //reading the hedges
  inc_read((hedge->num_inputs+1)*sizeof(edge)); //edge(in_mat_id,...)
  inc_read((hedge->num_inputs+1)*2*sizeof(long));
  // inc_read(hedge->size()); //reading the hedges
  double sum=0;
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
    
  }else if(hedge->factor_function == 1010){ // Softmax Loss
    double sum_y=0;
    for(int i=0; i<hedge->num_inputs; i++){
      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long offset=in_x*p_model->network->variables[in_mat_id].num_cols+in_y;
      double current_value=p_model->values[start_ind+offset];
      sum_y+=current_value;
      inc_add(1);
    }
    for(int i=0; i<hedge->num_inputs; i++){
      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long offset=in_x*p_model->network->variables[in_mat_id].num_cols+in_y;
      double init_value=p_model->network->variables[in_mat_id].init_value[offset];
      double current_value=p_model->values[start_ind+offset];
      // p_model->network->variables[in_mat_id].fid=i;
      if(abs(init_value-i)<EPS){
	      p_model->current_grads[start_ind+offset]=-(sum_y-current_value)/(current_value*sum_y);
	      p_model->current_grads[start_ind+offset]/=p_model->num_input;
	      inc_add(2);
      }else{
	      p_model->current_grads[start_ind+offset]=1.0/sum_y;
	      p_model->current_grads[start_ind+offset]/=p_model->num_input;
      }
      inc_write(2*sizeof(p_model->current_grads[start_ind+offset]));
      inc_add(1);
      inc_div(2);
    }
  }else if(hedge->factor_function == 1000){ // Conv
    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
    // sum = n_model->values[start_ind+offset];


    double grad1 = 1.0; 
    grad1 = grad1 * n_model->current_grads[start_ind+offset];
    inc_mul(1);

    for(int i=0; i<hedge->num_inputs; i++){
      weight * new_weight_i = &p_model->new_weights[hedge->start_ind[i].weight_id];

      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long nCols=p_model->network->variables[in_mat_id].num_cols;
      const long n_rows_in_mat = new_weight_i->num_rows;
      const long n_cols_in_mat = new_weight_i->num_cols;
      const double * p_row_in_mat = &p_model->values[start_ind + (in_x)*nCols + in_y];
      double * p_row_grads = &p_model->current_grads[start_ind + (in_x)*nCols + in_y];
      double * p_row_weights = &new_weight_i->values[0];
      int c;
      for(int r=0; r<n_rows_in_mat; r++){
        inc_add(2*n_cols_in_mat);
        inc_mul(3*n_cols_in_mat);
        c=0;
        for(c=0; c+4<n_cols_in_mat; c+=5){
          p_row_grads[c] += grad1 * p_row_weights[c];
          p_row_grads[c+1] += grad1 * p_row_weights[c+1];
          p_row_grads[c+2] += grad1 * p_row_weights[c+2];
          p_row_grads[c+3] += grad1 * p_row_weights[c+3];
          p_row_grads[c+4] += grad1 * p_row_weights[c+4];

          p_row_weights[c] -= learn_rate * grad1*p_row_in_mat[c];
          p_row_weights[c+1] -= learn_rate * grad1*p_row_in_mat[c+1];
          p_row_weights[c+2] -= learn_rate * grad1*p_row_in_mat[c+2];
          p_row_weights[c+3] -= learn_rate * grad1*p_row_in_mat[c+3];
          p_row_weights[c+4] -= learn_rate * grad1*p_row_in_mat[c+4];
        }
        if(n_cols_in_mat%5!=0){
          for(;c<n_cols_in_mat;c++){
            p_row_grads[c] += grad1 * p_row_weights[c];
            p_row_weights[c] -= learn_rate * grad1*p_row_in_mat[c];
          }
        }
        p_row_in_mat += nCols;
        p_row_weights += n_cols_in_mat;
        p_row_grads += nCols;
      }
      new_weight_i->bias-=learn_rate * grad1; //TODO Check
      inc_add(1);
      inc_mul(1);
      // inc_write(1*sizeof(double));
    }

  }else if(hedge->factor_function == 1001){ // Average
	  sum = 0.0;
    for(int i=0; i<hedge->num_inputs; i++){
        long in_mat_id = hedge->start_ind[i].in_mat_id;
        long in_x=hedge->start_ind[i].in_center_x;
        long in_y=hedge->start_ind[i].in_center_y;
        const weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];

        long start_ind=p_model->network->variables[in_mat_id].start_ind;
        long nCols=p_model->network->variables[in_mat_id].num_cols;
       
        const long n_rows_in_mat = weight_i->num_rows;
        const long n_cols_in_mat = weight_i->num_cols;
        const double * p_row_in_mat = &p_model->values[start_ind + (in_x)*nCols + in_y];
        const double * p_row_weights = &weight_i->values[0];

        // TODO: Change to SIMD
        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
        int c;
        for(int r=0; r<n_rows_in_mat; r++){
          inc_add(n_cols_in_mat);
          inc_mul(n_cols_in_mat);
          c = 0;
          for(c=0; c+4<n_cols_in_mat; c+=5){   
            sum1 += p_row_weights[c+0] * p_row_in_mat[c+0];
            sum2 += p_row_weights[c+1] * p_row_in_mat[c+1];
            sum3 += p_row_weights[c+2] * p_row_in_mat[c+2];
            sum4 += p_row_weights[c+3] * p_row_in_mat[c+3];
            sum5 += p_row_weights[c+4] * p_row_in_mat[c+4];
          }
          // show(n_cols_in_mat);
          if(n_cols_in_mat%5!=0){
            for(;c<n_cols_in_mat;c++){
             sum += p_row_weights[c] * p_row_in_mat[c];
            }
          } // TODO: good candidate for Just-in-time (JIT) compilation

          p_row_in_mat += nCols;
          p_row_weights += n_cols_in_mat;
       }
       sum += sum1 + sum2 + sum3 + sum4 + sum5;

      if(i==0){
        sum += weight_i->bias;
        inc_add(1);
      }
    }


	  cnn_layer_model* n_model=p_model->next;
	  long out_id=hedge->out_mat_id;
	  long out_x=hedge->out_center_x;
	  long out_y=hedge->out_center_y;
	  long start_ind=n_model->network->variables[out_id].start_ind;
	  long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
	  // inc_read(n_model->network->variables[out_id].size());


	  double grad1 = (1.0/(1.0+exp(-sum))) *  (1.0-(1.0/(1.0+exp(-sum)))); //SIGMOID
	  inc_add(5);
	  inc_mul(1);
	  inc_div(2);
	  inc_com(2);

	  // double grad1 = 1- ((exp(sum)-exp(-sum))/(exp(sum) + exp(-sum))) * ((exp(sum)-exp(-sum))/(exp(sum) + exp(-sum))); //TANH
	  grad1 = grad1 * n_model->current_grads[start_ind+offset];
	  // inc_read(sizeof(n_model->current_grads[start_ind+offset]));

	  inc_mul(1);

    for(int i=0; i<hedge->num_inputs; i++){
      weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];

      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long nCols=p_model->network->variables[in_mat_id].num_cols;
      const long n_rows_in_mat = weight_i->num_rows;
      const long n_cols_in_mat = weight_i->num_cols;
      double * p_row_grads = &p_model->current_grads[start_ind + (in_x)*nCols + in_y];
      double * p_row_weights = &weight_i->values[0];

      for(int r=0; r<n_rows_in_mat; r++){
        inc_add(2*n_cols_in_mat);
        inc_mul(3*n_cols_in_mat);
        for(int c=0; c<n_cols_in_mat; c++){
          p_row_grads[c] += grad1 * p_row_weights[c];
        }
        p_row_weights += n_cols_in_mat;
        p_row_grads += nCols;
      }
    }

  }else if(hedge->factor_function == 1002){ // Max
	  double my_max = -1E99;
	  inc_add(1);
	  double max_i=0;
	  double max_r=0;
	  double max_c=0;
	  for(int i=0; i<hedge->num_inputs; i++){
		  const weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];
		  long in_mat_id = hedge->start_ind[i].in_mat_id;
		  long in_x=hedge->start_ind[i].in_center_x;
		  long in_y=hedge->start_ind[i].in_center_y;
		  long start_ind=p_model->network->variables[in_mat_id].start_ind;

		  for(int r=0; r<weight_i->num_rows; r++)
			  for(int c=0; c<weight_i->num_cols; c++){
				  long offset=(in_x+r)*p_model->network->variables[in_mat_id].num_cols+(in_y+c);
				  if(p_model->values[start_ind+offset]>my_max){
					  max_i=i;
					  max_c=c;
					  max_r=r;
					  my_max = p_model->values[start_ind+offset];
				  }
				  inc_add(1);
			  }
	  }

	  cnn_layer_model* n_model=p_model->next;
	  long out_id=hedge->out_mat_id;
	  long out_x=hedge->out_center_x;
	  long out_y=hedge->out_center_y;
	  long start_ind=n_model->network->variables[out_id].start_ind;
	  long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
	  // inc_read(n_model->network->variables[out_id].size());

	  double grad1 = (1.0/(1.0+exp(-my_max))) *  (1.0-(1.0/(1.0+exp(-my_max)))); //SIGMOID 
	  inc_add(5);
	  inc_mul(1);
	  inc_div(2);
	  inc_com(2);

	  // double grad1 = 1- ((exp(sum)-exp(-sum))/(exp(sum) + exp(-sum))) * ((exp(sum)-exp(-sum))/(exp(sum) + exp(-sum))); //TANH
	  grad1 = grad1 * n_model->current_grads[start_ind+offset];
	  // inc_read(sizeof(n_model->current_grads[start_ind+offset]));

	  inc_mul(1);

	  for(int i=0; i<hedge->num_inputs; i++){
		  weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];
		  long in_mat_id = hedge->start_ind[i].in_mat_id;
		  long in_x=hedge->start_ind[i].in_center_x;
		  long in_y=hedge->start_ind[i].in_center_y;
		  long start_ind=p_model->network->variables[in_mat_id].start_ind;
		  for(int r=0; r<weight_i->num_rows; r++)
			  for(int c=0; c<weight_i->num_cols; c++){
				  long offset=(in_x+r)*p_model->network->variables[in_mat_id].num_cols+(in_y+c);
				  if(max_r==r && max_c==c && max_i==i){
					  p_model->current_grads[start_ind+offset] += grad1;
					  inc_add(1);
					  inc_write(1*sizeof(double));
				  }
			  }
	  }

  }else if(hedge->factor_function == 1005){ // Hidden
	  sum = 0.0;
    for(int i=0; i<hedge->num_inputs; i++){
        long in_mat_id = hedge->start_ind[i].in_mat_id;
        long in_x=hedge->start_ind[i].in_center_x;
        long in_y=hedge->start_ind[i].in_center_y;
        const weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];

        long start_ind=p_model->network->variables[in_mat_id].start_ind;
        long nCols=p_model->network->variables[in_mat_id].num_cols;
       
        const long n_rows_in_mat = weight_i->num_rows;
        const long n_cols_in_mat = weight_i->num_cols;
        const double * p_row_in_mat = &p_model->values[start_ind + (in_x)*nCols + in_y];
        const double * p_row_weights = &weight_i->values[0];

        // TODO: Change to SIMD
        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
        int c;
        for(int r=0; r<n_rows_in_mat; r++){
          inc_add(n_cols_in_mat);
          inc_mul(n_cols_in_mat);
          c = 0;
          for(c=0; c+4<n_cols_in_mat; c+=5){   
            sum1 += p_row_weights[c+0] * p_row_in_mat[c+0];
            sum2 += p_row_weights[c+1] * p_row_in_mat[c+1];
            sum3 += p_row_weights[c+2] * p_row_in_mat[c+2];
            sum4 += p_row_weights[c+3] * p_row_in_mat[c+3];
            sum5 += p_row_weights[c+4] * p_row_in_mat[c+4];
          }
          // show(n_cols_in_mat);
          if(n_cols_in_mat%5!=0){
            for(;c<n_cols_in_mat;c++){
             sum += p_row_weights[c] * p_row_in_mat[c];
            }
          } // TODO: good candidate for Just-in-time (JIT) compilation

          p_row_in_mat += nCols;
          p_row_weights += n_cols_in_mat;
       }
       sum += sum1 + sum2 + sum3 + sum4 + sum5;

      if(i==0){
        sum += weight_i->bias;
        inc_add(1);
      }
    }


	  // show(sum);

	  cnn_layer_model* n_model=p_model->next;
	  long out_id=hedge->out_mat_id;
	  long out_x=hedge->out_center_x;
	  long out_y=hedge->out_center_y;
	  // show(out_id);
	  long start_ind=n_model->network->variables[out_id].start_ind;
	  long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
	  // inc_read(n_model->network->variables[out_id].size());

	  // double grad1 = (1.0/(1.0+exp(-sum))) *  (1.0-(1.0/(1.0+exp(-sum)))); //SIGMOID 
	  double grad1 = 1- ((exp(sum)-exp(-sum))/(exp(sum) + exp(-sum))) * ((exp(sum)-exp(-sum))/(exp(sum) + exp(-sum))); //TANH
	  inc_add(9);
	  inc_mul(1);
	  inc_div(2);
	  grad1 = grad1 * n_model->current_grads[start_ind+offset];
	  // inc_read(sizeof(n_model->current_grads[start_ind+offset]));

	  inc_mul(1);
    for(int i=0; i<hedge->num_inputs; i++){
      weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];
      weight * new_weight_i = &p_model->new_weights[hedge->start_ind[i].weight_id];

      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long nCols=p_model->network->variables[in_mat_id].num_cols;
      const long n_rows_in_mat = new_weight_i->num_rows;
      const long n_cols_in_mat = new_weight_i->num_cols;
      const double * p_row_in_mat = &p_model->values[start_ind + (in_x)*nCols + in_y];
      double * p_row_grads = &p_model->current_grads[start_ind + (in_x)*nCols + in_y];
      double * p_row_new_weights = &new_weight_i->values[0];
      double * p_row_weights = &weight_i->values[0];

      for(int r=0; r<n_rows_in_mat; r++){
        inc_add(2*n_cols_in_mat);
        inc_mul(3*n_cols_in_mat);
        for(int c=0; c<n_cols_in_mat; c++){
          p_row_grads[c] += grad1 * p_row_weights[c];
          p_row_new_weights[c] -= learn_rate * grad1*p_row_in_mat[c];//+reg_rate*weight_i->values[r*network.weights[id].num_cols+c];          
        }
        p_row_in_mat += nCols;
        p_row_weights += n_cols_in_mat;
        p_row_new_weights += n_cols_in_mat;
        p_row_grads += nCols;

      }
      new_weight_i->bias-=learn_rate * grad1; //TODO Check
      inc_add(1);
      inc_mul(1);
      // inc_write(1*sizeof(double));
    }
  }else if(hedge->factor_function == 1020){ // softmax
	  cnn_layer_model* n_model=p_model->next;
	  long out_id=hedge->out_mat_id;
	  long out_x=hedge->out_center_x;
	  long out_y=hedge->out_center_y;
	  long start_ind=n_model->network->variables[out_id].start_ind;
	  long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
	  // inc_read(n_model->network->variables[out_id].size());

	  double grad1 = n_model->values[start_ind+offset];

	  grad1 = grad1 * n_model->current_grads[start_ind+offset];
	  // inc_read(sizeof(n_model->current_grads[start_ind+offset]));

	  inc_mul(1);

	  for(int i=0; i<hedge->num_inputs; i++){
      weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];
      weight * new_weight_i = &p_model->new_weights[hedge->start_ind[i].weight_id];

      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long nCols=p_model->network->variables[in_mat_id].num_cols;
      const long n_rows_in_mat = new_weight_i->num_rows;
      const long n_cols_in_mat = new_weight_i->num_cols;
      const double * p_row_in_mat = &p_model->values[start_ind + (in_x)*nCols + in_y];
      double * p_row_grads = &p_model->current_grads[start_ind + (in_x)*nCols + in_y];
      double * p_row_new_weights = &new_weight_i->values[0];
      double * p_row_weights = &weight_i->values[0];

      for(int r=0; r<n_rows_in_mat; r++){
        inc_add(2*n_cols_in_mat);
        inc_mul(3*n_cols_in_mat);
        for(int c=0; c<n_cols_in_mat; c++){
          p_row_grads[c] += grad1 * p_row_weights[c];
          p_row_new_weights[c] -= learn_rate * grad1*p_row_in_mat[c];//+reg_rate*weight_i->values[r*network.weights[id].num_cols+c];          
        }
        p_row_in_mat += nCols;
        p_row_weights += n_cols_in_mat;
        p_row_new_weights += n_cols_in_mat;
        p_row_grads += nCols;

      }
      new_weight_i->bias-=learn_rate * grad1; //TODO Check
      inc_add(1);
      inc_mul(1);
      // inc_write(1*sizeof(double));
    }
  }else{
	  std::cout << "FUNCTION ID " << hedge->factor_function << " NOT DEFINED!" << std::endl;
	  assert(false);
  }
  // show(end);
  // show(seconds);
  return 0.0;
}

 bool print=1;

 double forward_propogate(const SparseVector<double>* const ex, cnn_layer_model* const p_model){
   // cout << "Forward propogating\n";
    long hedge_ind=ex->p[0];
    double sum = 0.0;

    // show(hedge_ind);
    hyper_edge * hedge=&p_model->network->hedges[hedge_ind];
    inc_read(sizeof(ex->p)); // reading p[0]
    inc_read(sizeof(hyper_edge)); //reading the hedges
    inc_read((hedge->num_inputs+1)*sizeof(edge)); //edge(in_mat_id,...)
    inc_read((hedge->num_inputs+1)*2*sizeof(long));
    // inc_write(1*sizeof(double));

    if(hedge->factor_function == 1000){ // Conv
      sum = 0.0;
      //return sum;
      for(int i=0; i<hedge->num_inputs; i++){
        long in_mat_id = hedge->start_ind[i].in_mat_id;
        long in_x=hedge->start_ind[i].in_center_x;
        long in_y=hedge->start_ind[i].in_center_y;
        const weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];

        long start_ind=p_model->network->variables[in_mat_id].start_ind;
        long nCols=p_model->network->variables[in_mat_id].num_cols;
       
        const long n_rows_in_mat = weight_i->num_rows;
        const long n_cols_in_mat = weight_i->num_cols;
        const double * p_row_in_mat = &p_model->values[start_ind + (in_x)*nCols + in_y];
        const double * p_row_weights = &weight_i->values[0];

        // TODO: Change to SIMD
        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
        int c;
        for(int r=0; r<n_rows_in_mat; r++){
          inc_add(n_cols_in_mat);
          inc_mul(n_cols_in_mat);
          c = 0;
          for(c=0; c+4<n_cols_in_mat; c+=5){	 
            sum1 += p_row_weights[c+0] * p_row_in_mat[c+0];
            sum2 += p_row_weights[c+1] * p_row_in_mat[c+1];
            sum3 += p_row_weights[c+2] * p_row_in_mat[c+2];
            sum4 += p_row_weights[c+3] * p_row_in_mat[c+3];
            sum5 += p_row_weights[c+4] * p_row_in_mat[c+4];
          }
          // show(n_cols_in_mat);
          if(n_cols_in_mat%5!=0){
            for(;c<n_cols_in_mat;c++){
             sum += p_row_weights[c] * p_row_in_mat[c];
            }
          } // TODO: good candidate for Just-in-time (JIT) compilation

          p_row_in_mat += nCols;
          p_row_weights += n_cols_in_mat;
       }
       sum += sum1 + sum2 + sum3 + sum4 + sum5;

      if(i==0){
        sum += weight_i->bias;
        inc_add(1);
      }
    }

    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
    
    // inc_read(sizeof(long)*2);
    // inc_read(sizeof(p_model->next)+sizeof(n_model->network));
    //sum+=n_model->values[start_ind+offset];
    n_model->values[start_ind+offset] = sum;
    // inc_write(1*sizeof(double));
  }else if(hedge->factor_function == 1001){ // Ave
    sum = 0.0;

    for(int i=0; i<hedge->num_inputs; i++){
      const weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];
      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      for(int r=0; r<weight_i->num_rows; r++)
        for(int c=0; c<weight_i->num_cols; c++){
          long offset=(in_x+r)*p_model->network->variables[in_mat_id].num_cols+(in_y+c);
          sum += weight_i->values[r*weight_i->num_cols+c] * p_model->values[start_ind+offset];
          inc_add(1);
          inc_mul(1);
        }
    }

    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
    n_model->values[start_ind+offset] = (1.0)/(1.0 + exp(-sum)); //Sigmoid
    inc_add(2);
    inc_div(1);
  }else if(hedge->factor_function == 1002){ // Max
    double my_max=-1E99;
    inc_add(1);
    for(int i=0; i<hedge->num_inputs; i++){
      const weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];
      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;

      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long nCols=p_model->network->variables[in_mat_id].num_cols;


      const long n_rows_in_mat = weight_i->num_rows;
      const long n_cols_in_mat = weight_i->num_cols;
      const double * p_row_in_mat = &p_model->values[start_ind + (in_x)*nCols + in_y];
      const double * p_row_weights = &weight_i->values[0];


      for(int r=0; r<n_rows_in_mat; r++){
        for(int c=0; c<n_cols_in_mat; c++){  
          if(p_row_in_mat[c]>my_max){
            my_max=p_row_in_mat[c];
          }
          inc_div(1);
        }
        p_row_in_mat += nCols;
      }
    }

    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
    n_model->values[start_ind+offset] = (1.0)/(1.0 + exp(-my_max)); //Sigmoid
    inc_write(1*sizeof(double));
    inc_add(2);
    inc_div(1);
    inc_com(1);
    // n_model->values[start_ind+offset] = (exp(sum)-exp(-sum))/(exp(sum) + exp(-sum)); //Tanh
    // n_model->values[start_ind+offset] = (exp(sum)-exp(-sum))/(exp(sum) + exp(-sum)) + tanh_bias*sum; //Tanh

  }else if(hedge->factor_function == 1005){ // Hidden
    sum = 0.0;
    for(int i=0; i<hedge->num_inputs; i++){
        long in_mat_id = hedge->start_ind[i].in_mat_id;
        long in_x=hedge->start_ind[i].in_center_x;
        long in_y=hedge->start_ind[i].in_center_y;
        const weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];

        long start_ind=p_model->network->variables[in_mat_id].start_ind;
        long nCols=p_model->network->variables[in_mat_id].num_cols;
       
        const long n_rows_in_mat = weight_i->num_rows;
        const long n_cols_in_mat = weight_i->num_cols;
        const double * p_row_in_mat = &p_model->values[start_ind + (in_x)*nCols + in_y];
        const double * p_row_weights = &weight_i->values[0];

        // TODO: Change to SIMD
        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
        int c;
        for(int r=0; r<n_rows_in_mat; r++){
          inc_add(n_cols_in_mat);
          inc_mul(n_cols_in_mat);
          c = 0;
          for(c=0; c+4<n_cols_in_mat; c+=5){   
            sum1 += p_row_weights[c+0] * p_row_in_mat[c+0];
            sum2 += p_row_weights[c+1] * p_row_in_mat[c+1];
            sum3 += p_row_weights[c+2] * p_row_in_mat[c+2];
            sum4 += p_row_weights[c+3] * p_row_in_mat[c+3];
            sum5 += p_row_weights[c+4] * p_row_in_mat[c+4];
          }
          // show(n_cols_in_mat);
          if(n_cols_in_mat%5!=0){
            for(;c<n_cols_in_mat;c++){
             sum += p_row_weights[c] * p_row_in_mat[c];
            }
          } // TODO: good candidate for Just-in-time (JIT) compilation

          p_row_in_mat += nCols;
          p_row_weights += n_cols_in_mat;
       }
       sum += sum1 + sum2 + sum3 + sum4 + sum5;

      if(i==0){
        sum += weight_i->bias;
        inc_add(1);
      }
    }

    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
    // n_model->values[start_ind+offset] = (1.0)/(1.0 + exp(-sum)); //Sigmoid
    n_model->values[start_ind+offset] = (exp(sum)-exp(-sum))/(exp(sum) + exp(-sum)); //Tanh
    inc_write(1*sizeof(double));
    inc_add(4);
    inc_div(1);
    inc_com(4);

    // n_model->values[start_ind+offset] = (exp(sum)-exp(-sum))/(exp(sum) + exp(-sum)) + tanh_bias*sum; //Tanh
  }else if(hedge->factor_function == 1020){ // softmax
    sum = 0.0;

    for(int i=0; i<hedge->num_inputs; i++){
        long in_mat_id = hedge->start_ind[i].in_mat_id;
        long in_x=hedge->start_ind[i].in_center_x;
        long in_y=hedge->start_ind[i].in_center_y;
        const weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];

        long start_ind=p_model->network->variables[in_mat_id].start_ind;
        long nCols=p_model->network->variables[in_mat_id].num_cols;
       
        const long n_rows_in_mat = weight_i->num_rows;
        const long n_cols_in_mat = weight_i->num_cols;
        const double * p_row_in_mat = &p_model->values[start_ind + (in_x)*nCols + in_y];
        const double * p_row_weights = &weight_i->values[0];

        // TODO: Change to SIMD
        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
        int c;
        for(int r=0; r<n_rows_in_mat; r++){
          inc_add(n_cols_in_mat);
          inc_mul(n_cols_in_mat);
          c = 0;
          for(c=0; c+4<n_cols_in_mat; c+=5){   
            sum1 += p_row_weights[c+0] * p_row_in_mat[c+0];
            sum2 += p_row_weights[c+1] * p_row_in_mat[c+1];
            sum3 += p_row_weights[c+2] * p_row_in_mat[c+2];
            sum4 += p_row_weights[c+3] * p_row_in_mat[c+3];
            sum5 += p_row_weights[c+4] * p_row_in_mat[c+4];
          }
          // show(n_cols_in_mat);
          if(n_cols_in_mat%5!=0){
            for(;c<n_cols_in_mat;c++){
             sum += p_row_weights[c] * p_row_in_mat[c];
            }
          } // TODO: good candidate for Just-in-time (JIT) compilation

          p_row_in_mat += nCols;
          p_row_weights += n_cols_in_mat;
       }
       sum += sum1 + sum2 + sum3 + sum4 + sum5;

      if(i==0){
        sum += weight_i->bias;
        inc_add(1);
      }
    }

    cnn_layer_model* n_model=p_model->next;
    long out_id=hedge->out_mat_id;
    long out_x=hedge->out_center_x;
    long out_y=hedge->out_center_y;
    long start_ind=n_model->network->variables[out_id].start_ind;
    long offset=out_x*n_model->network->variables[out_id].num_cols+out_y;
    n_model->values[start_ind+offset] = exp(sum);
    inc_write(1*sizeof(double));
    inc_com(1);
  }else{
    std::cout << "FUNCTION ID " << hedge->factor_function << " NOT DEFINED!" << std::endl;
    assert(false);
  }
  return sum;
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

  }else if(hedge->factor_function == 1010){ // softmax error
    double denom=0;
    double numer=0;
    double init_value;
    for(int i=0; i<hedge->num_inputs; i++){
      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long offset=in_x*p_model->network->variables[in_mat_id].num_cols+in_y;
      init_value=p_model->network->variables[in_mat_id].init_value[offset];
      double current_value=p_model->values[start_ind+offset];
      if(abs(init_value-i)<EPS)
        numer=current_value;
      denom+=current_value;
    }
    // double l2_norm=0;
    // for(int i=0; i<p_model->network->num_weights; i++){
    //   weight * weight_i = &p_model->weights[i];
    //   for(int r=0; r<weight_i->num_rows; r++)
    //     for(int c=0; c<weight_i->num_cols; c++){
    //       l2_norm += (weight_i->values[r*weight_i->num_cols+c] * weight_i->values[r*weight_i->num_cols+c]);
    //     }
    // }
    double prob=numer/denom;
    std::cout.precision(10);
    // std::cout << "Prob  " << prob << " for class: " << init_value << std::endl;

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
    double init_value=-1;
    double max_value=-1;
    long max_index=0;
    for(int i=0; i<hedge->num_inputs; i++){
      long in_mat_id = hedge->start_ind[i].in_mat_id;
      long in_x=hedge->start_ind[i].in_center_x;
      long in_y=hedge->start_ind[i].in_center_y;
      long start_ind=p_model->network->variables[in_mat_id].start_ind;
      long offset=in_x*p_model->network->variables[in_mat_id].num_cols+in_y;
      init_value=p_model->network->variables[in_mat_id].init_value[offset];
      double current_value=p_model->values[start_ind+offset];
      if(abs(i-init_value)<EPS)
        numer=current_value;
      if(current_value>max_value){
        max_value=current_value;
        max_index=i;
      }
      denom+=current_value;
    } 

    // std::cout << "max_value: " << max_value/denom << " true value: " << numer/denom << " for class: " << init_value << std::endl;

    if(abs(init_value-max_index)>EPS)
      return 100.0/p_model->num_input;
    else return 0;
    // return loss;
  }else{
    std::cout << "FUNCTION ID " << hedge->factor_function << " NOT DEFINED!" << std::endl;
    assert(false);
  }
  return 0.0;
}

double update(const SparseVector<double>* const ex, cnn_layer_model* const p_model){
  long hedge_ind=ex->p[0];
  hyper_edge* hedge=&p_model->network->hedges[hedge_ind];

  for(int i=0; i<hedge->num_inputs; i++){
    weight * weight_i = &p_model->weights[hedge->start_ind[i].weight_id];
    weight * new_weight_i = &p_model->new_weights[hedge->start_ind[i].weight_id];
    for(int r=0; r<weight_i->num_rows*weight_i->num_cols; r++)
      weight_i->values[r]=new_weight_i->values[r];
    weight_i->bias=new_weight_i->bias;
  }
  return 0.0;
}

template<ModelReplType MODELREPL, DataReplType DATAREPL>
double cnn_sparse(neural_network &network){

  int num_layer=7;
  // cout.precision(15);
  
  long * nexp=new long[num_layer];
  long * nfeat=new long[num_layer];
  double ** examples = new double * [num_layer];
  long ** cols = new long * [num_layer];
  long ** rows = new long * [num_layer];
  cnn_layer_model * models= new cnn_layer_model [num_layer];
  unsigned int * f_handle_f_prop=new unsigned int [num_layer];
  unsigned int * f_handle_b_prop=new unsigned int [num_layer];
  unsigned int * f_handle_update=new unsigned int [num_layer];

  unsigned int f_handle_error;
  unsigned int f_handle_v_error;


  long * var_size_layer=new long[num_layer];
  long * hedges_size_layer=new long[num_layer];
  long * num_var_layer=new long[num_layer];
  long * num_hedges_layer=new long[num_layer];
  for(int l=0; l<num_layer; l++){
    var_size_layer[l]=0;
    hedges_size_layer[l]=0;
    num_var_layer[l]=0;
    num_hedges_layer[l]=0;
  }



  SparseDimmWitted<double, cnn_layer_model, MODELREPL, DATAREPL, DW_ACCESS_ROW> ** dw_l=
    new SparseDimmWitted<double, cnn_layer_model, MODELREPL, DATAREPL, DW_ACCESS_ROW> *[num_layer];


  for(int l=0; l<num_layer; l++){
    //calculate number of variables in layer i(nexp)
    nfeat[l]=0;
    for(int i=0; i<network.num_vars; i++){
      if(network.variables[i].layer==l){
        nfeat[l]++;
      }
    }

    //calculate number of hedges with input variable in layer i (nexp)
    nexp[l]=0;
    for(int i=0; i<network.num_hedges; i++){
      for(int j=0; j<network.hedges[i].num_inputs; j++){
        if(network.variables[network.hedges[i].start_ind[j].in_mat_id].layer==l){
          nexp[l]++;
          network.hedges[i].layer=l; //initialize the layer of each hedge
          break;
        }
      }
    }
    // show(nexp[l]*(nfeat[l]+2));
    // examples[l] = new double[nexp[l]*(nfeat[l]+2)];
    // cols[l] = new long[nexp[l]*(nfeat[l]+2)];
    examples[l] = new double[nexp[l]];
    cols[l] = new long[nexp[l]];
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
        // network.hedges[i].start_ind=col_i;

        //Examples=[hedge_id,in_id1,in_id2,...,in_idn,out_id]
        examples[l][col_i]=i; //hedge id
        cols[l][col_i]=0;
        col_i++;

        // for(int j=0; j<network.hedges[i].in_mat_ids.size(); j++){
        //   examples[l][col_i]=network.hedges[i].in_mat_ids[j]; //Input -> 1
        //   long ind=network.hedges[i].in_mat_ids[j];
        //   // show(ind);
        //   cols[l][col_i]=network.variables[ind].var_layer_ind+1;
        //   col_i++;
        // }

        // examples[l][col_i]=network.hedges[i].out_mat_id; //Output -> 2
        // cols[l][col_i]=nfeat[l]+1;
        // col_i++;
      }
    }
    // show(col_i);



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
          }
      }
    models[l].size=num_var[l];
    models[l].num_input=nfeat[0];

    double fan_in0 = 1*5*5; //fmap_ind*filter_nrow*filter_ncols
    double fan_out0 = 4*5*5/(2*2); //(fmap_out*filter_nrow*filter_ncols)/(pool_nrows*pool_ncols)
    double W_bound0 = sqrt(6.0 / (fan_in0 + fan_out0));
    show(W_bound0);

    double fan_in2 = 4*5*5; //fmap_ind*filter_nrow*filter_ncols
    double fan_out2 = 6*5*5/(2*2); //(fmap_out*filter_nrow*filter_ncols)/(pool_nrows*pool_ncols)
    double W_bound2 = sqrt(6.0 / (fan_in2 + fan_out2));
    show(W_bound2);

    double fan_in4 = 6*4*4; //fmap_ind*filter_nrow*filter_ncols
    double fan_out4 = 20; //(fmap_out*filter_nrow*filter_ncols)/(pool_nrows*pool_ncols)
    double W_bound4 = sqrt(6.0 / (fan_in4 + fan_out4));
    show(W_bound4);
    uniform_real_distribution<double> distribution0(-W_bound0,W_bound0);
    uniform_real_distribution<double> distribution2(-W_bound2,W_bound2);
    uniform_real_distribution<double> distribution4(-W_bound4,W_bound4);
    default_random_engine generator(21457);
    //Init weights
    models[l].weights=new weight[network.num_weights];
    models[l].new_weights=new weight[network.num_weights];

    for(long i=0; i<network.num_weights; i++){
      models[l].weights[i].weight_id=i;
      models[l].weights[i].is_fixed=network.weights[i].is_fixed;
      models[l].new_weights[i].weight_id=i;
      models[l].new_weights[i].is_fixed=network.weights[i].is_fixed;

      models[l].weights[i].num_rows=network.weights[i].num_rows;
      models[l].weights[i].num_cols=network.weights[i].num_cols;
      models[l].new_weights[i].num_rows=network.weights[i].num_rows;
      models[l].new_weights[i].num_cols=network.weights[i].num_cols;

      models[l].weights[i].bias=0;
      models[l].new_weights[i].bias=0;
      models[l].weights[i].values=new double [network.weights[i].num_rows*network.weights[i].num_cols];
      models[l].new_weights[i].values=new double [network.weights[i].num_rows*network.weights[i].num_cols];
      long nRows=models[l].weights[i].num_rows;
      long nCols=models[l].weights[i].num_cols;
      // show(network.weights[i].num_rows);
      for(int offset=0; offset<nRows*nCols; offset++){
        // uniform_real_distribution<double> distribution(-1,1);
        if(l==0){
          double number = distribution0(generator);
          do
            number = distribution0(generator);
          while ((number<-5)&&(number>5));
          models[l].weights[i].values[offset]=number;
          models[l].new_weights[i].values[offset]=number;
        }else if(l==2){
          double number = distribution2(generator);
          do
            number = distribution2(generator);
          while ((number<-5)&&(number>5));
          models[l].weights[i].values[offset]=number;
          models[l].new_weights[i].values[offset]=number;

          // models[l].weights[i].values[j][k]=0.1;
          // models[l].new_weights[i].values[j][k]=0.1;
        }else if(l==4){
          double number = distribution4(generator);
          do
            number = distribution4(generator);
          while ((number<-5)&&(number>5));
          models[l].weights[i].values[offset]=number;
          models[l].new_weights[i].values[offset]=number;

          // models[l].weights[i].values[j][k]=0.1;
          // models[l].new_weights[i].values[j][k]=0.1;
        }else{
          models[l].weights[i].values[offset]=0;
          models[l].new_weights[i].values[offset]=0;
        }
        if(models[l].weights[i].is_fixed==1){
            models[l].weights[i].values[offset]=network.weights[i].initial_value;
            models[l].new_weights[i].values[offset]=network.weights[i].initial_value;
        }
      }
    }
    int * w_mark[3]; //layer,out_feature_map,order
    for(int i=0; i<3; i++){
      w_mark[i]=new int [network.num_weights];
      for(int j=0; j<network.num_weights; j++)
        w_mark[i][j]=-1;
    }

    // if(l==0){
    //   ifstream fin("weights");
    //   for(int ll=0; ll<num_layer-1; ll++){
    //     if(ll==1 || ll==3 || ll==5 )
    //       continue;
    //     for(int v=0; v<network.num_vars; v++){
    //       if(network.variables[v].layer==ll+1){
    //         for(long i=0; i<network.num_hedges; i++){
    //           if(network.hedges[i].out_mat_id==v){
    //             for(long j=0; j<network.hedges[i].num_inputs; j++){
    //               long id=network.hedges[i].start_ind[j].weight_id;
    //               if(w_mark[0][id]==-1){
    //                 w_mark[0][id]=1;
    //                 if(ll==4){
    //                   for(int r=0; r<network.weights[id].num_rows; r++){
    //                     for(int c=0; c<network.weights[id].num_cols; c++){
    //                       double w=-1;
    //                       fin >> w;
    //                       // show(w);
    //                       models[l].weights[id].values[r*network.weights[id].num_cols+c]=w;
    //                       models[l].new_weights[id].values[r*network.weights[id].num_cols+c]=w;
    //                       if(models[l].weights[id].is_fixed==1){
    //                           models[l].weights[id].values[r*network.weights[id].num_cols+c]=network.weights[id].initial_value;
    //                           models[l].new_weights[id].values[r*network.weights[id].num_cols+c]=network.weights[id].initial_value;
    //                       }
    //                     }
    //                   }
    //                 }
    //                 else
    //                   for(int r=network.weights[id].num_rows-1; r>=0; r--){
    //                     for(int c=network.weights[id].num_cols-1; c>=0; c--){
    //                       double w=-1;
    //                       fin >> w;
    //                       models[l].weights[id].values[r*network.weights[id].num_cols+c]=w;
    //                       models[l].new_weights[id].values[r*network.weights[id].num_cols+c]=w;
    //                       if(models[l].weights[id].is_fixed==1){
    //                           models[l].weights[id].values[r*network.weights[id].num_cols+c]=network.weights[id].initial_value;
    //                           models[l].new_weights[id].values[r*network.weights[id].num_cols+c]=network.weights[id].initial_value;
    //                       }
    //                     }
    //                   }
    //               }
    //             }
    //           }
    //         }
    //       }
    //     }
    //   }
    // }
    // else if(l<5)
    //   for(long i=0; i<network.num_weights; i++){
    //     for(int j=0; j<network.weights[i].num_rows; j++){
    //       for(int k=0; k<network.weights[i].num_cols; k++){
    //         models[l].weights[i].values[j*network.weights[i].num_cols+k]=models[0].weights[i].values[j*network.weights[i].num_cols+k];
    //         models[l].new_weights[i].values[j*network.weights[i].num_cols+k]=models[0].weights[i].values[j*network.weights[i].num_cols+k];
    //         if(models[l].weights[i].is_fixed==1){
    //             models[l].weights[i].values[j*network.weights[i].num_cols+k]=network.weights[i].initial_value;
    //             models[l].new_weights[i].values[j*network.weights[i].num_cols+k]=network.weights[i].initial_value;
    //         }
    //       }
    //     }
    //   }

   
    //Init next/prev pointers to the next/prev layer models
    if(l!=0)
      models[l].prev=&models[l-1];
    if(l!=num_layer-1)
      models[l].next=&models[l+1];
    models[l].layer=l;


    cout << "Initializing DW ...\n";

    //Initialize DW for layer l
    dw_l[l]=new SparseDimmWitted<double, cnn_layer_model, MODELREPL, DATAREPL, DW_ACCESS_ROW> 
      (examples[l], rows[l], cols[l], nexp[l], nfeat[l], col_i, &models[l]);

    cout << "Registering functions ...\n";
    //Register the functions
    if(l!=num_layer-1)
      f_handle_f_prop[l] = dw_l[l]->register_row(forward_propogate);
    f_handle_b_prop[l] = dw_l[l]->register_row(back_gradient);
    f_handle_update[l] = dw_l[l]->register_row(update);

  }
  f_handle_error = dw_l[num_layer-1]->register_row(error);
  f_handle_v_error=dw_l[num_layer-1]->register_row(v_error);
  double loss=0;
  double last_loss=-1;
  double validation_error=0;


  cout << "Data structure sizes\n";
  for(int i=0; i<network.num_vars; i++){
    var_size_layer[network.variables[i].layer]+=network.variables[i].size();
    num_var_layer[network.variables[i].layer]++;
  }
  long var_sum_size=0;
  long var_sum_num=0;
  for(int l=0; l<num_layer; l++){
    var_sum_size+=var_size_layer[l];
    var_sum_num+=num_var_layer[l];
    cout << "Size of variables of layer " << l << " for 1 image is: " << var_size_layer[l]/num_var_layer[0] << endl;
  }
  cout << "Total size of variables of all layers for 1 image is:  " << var_sum_size/num_var_layer[0] << endl;



  for(int i=0; i<network.num_hedges; i++){
    // hedges_size_layer[network.hedges[i].layer]+=network.hedges[i].size(); TODO
    num_hedges_layer[network.hedges[i].layer]++;
  }
  long hedge_sum_size=0;
  long hedge_sum_num=0;
  for(int l=0; l<num_layer-1; l++){
    hedge_sum_size+=hedges_size_layer[l];
    hedge_sum_num+=num_hedges_layer[l];
    cout << "Size of hedges from layer " << l << " to " << l+1 << " for 1 image is: " << hedges_size_layer[l]/num_var_layer[0] << endl;
  }
  cout << "Total size of hedges of all layers for 1 image is:  " << hedge_sum_size/num_var_layer[0] << endl;



  for(int i_epoch=0; i_epoch<1;i_epoch++){
    show(i_epoch);
    
    for(int l=0; l<num_layer-1; l++){
      cout << "************ Forward propogate layer : " << l << "************" << endl;

      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();
      
      reset_flop();
      reset_mem();

      uint64_t ncore=96;
      hazy::pmu::MSRAccess msr (ncore);
      hazy::pmu::UncorePMU pmu(hazy::pmu::PerfEvtSel0, 0, 0);
      pmu.OS(false);
      pmu.User(true);
      pmu.AnyThread(false);
      pmu.Enable(true);
      uint64_t response_flag=hazy::pmu::response::LLC_MISS_DRAM_1;
      uint64_t request_flag=hazy::pmu::request::ALL_DATA_1;
      //pmu.SetRequestRespondTest();
      //pmu.SetResponse(hazy::pmu::response::M_ANY);
      pmu.SetResponse(response_flag );
      // pmu.SetRequest(hazy::pmu::request::ALL );
      pmu.SetRequest(request_flag);
      pmu.ResetAll(msr);

      dw_l[l]->exec(f_handle_f_prop[l]);

      hazy::pmu::Registers reg = pmu.ReadAll(msr);
      for(int i=0; i<ncore; i++)
        if(i==90)
          cout << "(" << i << " ," << reg.regs[i]/num_var_layer[0] <<"), ";
      cout << endl;

      // inc_read(p_model->weights[hedge->start_ind[0].weight_id].num_rows*p_model->weights[hedge->start_ind[0].weight_id].num_cols*sizeof(double));
      // inc_read(p_model->network->variables[hedge->start_ind[0].in_mat_id].num_cols*p_model->network->variables[hedge->start_ind[0].in_mat_id].num_rows*sizeof(double));


      end = std::chrono::system_clock::now();      
      std::chrono::duration<double> elapsed_seconds = end-start;
      std::time_t end_time = std::chrono::system_clock::to_time_t(end);
      double elapsed_seconds_per_image=elapsed_seconds.count()/num_var_layer[0];
      std::cout << "elapsed time per image: " << elapsed_seconds_per_image << "s\n";
      // show(sizeof(double)*5*5*num_var_layer[l]/num_var_layer[0]*num_var_layer[l+1]/num_var_layer[0]); //weights
      inc_read(sizeof(double)*num_var[l]); //p_model->values
      inc_read(sizeof(long)*2*num_var_layer[l]); //num_cols,start_ind
      inc_read(sizeof(long)*2*num_var_layer[l+1]);  //num_cols,start_ind
      inc_write(sizeof(double)*num_var[l+1]); //n_model

      

      print_flop(num_var_layer[0]);
      print_flops(num_var_layer[0],elapsed_seconds_per_image);

      print_mem(num_var_layer[0]);
      print_mems(num_var_layer[0],elapsed_seconds_per_image);

      // cout<< "Values:\n";
      // int size;
      // for(int j=0; j<models[l].network->num_vars; j++)
      //   if(models[l].network->variables[j].layer==l)
      //     size=models[l].network->variables[j].num_rows*models[l].network->variables[j].num_cols;
      // show(size);
      // for(int i=0; i<models[l].size/size; i++){
      //   // if(i==0 || i==models[l].size/size-1)
      //     cout << "[";
      //   for(int j=0; j<size; j++)
      //     // if(i==0 || i==models[l].size/size-1)
      //       cout << models[l].values[i*size+j] << ", ";
      //   // if(i==0 || i==models[l].size/size-1)
      //     cout << "]" << endl;
      // }
      // cout << "Grads:\n";
      // // cout.precision(15);
      // for(int i=0; i<models[l].size/size; i++){
      //   if(i==0 || i==models[l].size/size-1)
      //     cout << "[";
      //   for(int j=0; j<size; j++)
      //     if(i==0 || i==models[l].size/size-1)
      //       cout << models[l].current_grads[i*size+j] << ", ";
      //   if(i==0 || i==models[l].size/size-1)
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

      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();

      reset_flop();
      reset_mem();

      // for(int i=0; i<network.num_weights; i++){
      //   inc_read(network.weights[i].mem_size());
      //   inc_write(network.weights[i].mem_size()*2);
      // }
      // inc_read(models[l].mem_size());
      dw_l[l]->exec(f_handle_b_prop[l]);
      dw_l[l]->exec(f_handle_update[l]);

      end = std::chrono::system_clock::now();      
      std::chrono::duration<double> elapsed_seconds = end-start;
      double elapsed_seconds_per_image=elapsed_seconds.count()/num_var_layer[0];
      std::cout << "elapsed time per image: " << elapsed_seconds_per_image << "s\n";
      inc_read(sizeof(double)*2*num_var[l]); //p_model->values & p_model->grads
      inc_read(sizeof(long)*2*num_var_layer[l]); //num_cols,start_ind
      inc_read(sizeof(long)*2*num_var_layer[l+1]);  //num_cols,start_ind
      inc_write(sizeof(double)*2*num_var[l+1]); //n_model


      print_flop(num_var_layer[0]);
      print_flops(num_var_layer[0],elapsed_seconds_per_image);

      print_mem(num_var_layer[0]);
      print_mems(num_var_layer[0],elapsed_seconds_per_image);

    }
    double sum1=0;

    for(int l=0; l<num_layer-1; l++){
      std::cout.precision(10);
      cout << "Layer " << l << " weights:" << endl;
      for(int i=0; i<models[l].network->num_hedges; i++)
        if(models[l].network->hedges[i].layer==l){
          const weight * weight_i = &models[l].weights[models[l].network->hedges[i].start_ind[0].weight_id];
          for(int j=0; j<weight_i->num_rows; j++){
            for(int k=0; k<weight_i->num_cols; k++){
              // std::cout.precision(50);
              int offset=j*weight_i->num_cols+k;
              cout << weight_i->values[offset] << " ";
            }
           cout<< endl;
          }
          // std::cout.precision(50);
          cout << "bias: " << weight_i->bias << endl;
          break;
        }
      cout << endl;
    }
    last_loss=loss;
  }
  return 0.0;
}


#endif
