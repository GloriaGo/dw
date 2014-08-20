#ifndef _NEURAL_MODEL_H
#define _NEURAL_MODEL_H

#include "io/cmd_parser.h"
#include <fstream>
#include <vector>
using namespace std;

# define bswap_64(x) \
     ((((x) & 0xff00000000000000ull) >> 56)                                   \
      | (((x) & 0x00ff000000000000ull) >> 40)                                 \
      | (((x) & 0x0000ff0000000000ull) >> 24)                                 \
      | (((x) & 0x000000ff00000000ull) >> 8)                                  \
      | (((x) & 0x00000000ff000000ull) << 8)                                  \
      | (((x) & 0x0000000000ff0000ull) << 24)                                 \
      | (((x) & 0x000000000000ff00ull) << 40)                                 \
      | (((x) & 0x00000000000000ffull) << 56))

// 16-bit big endian to little endian
#define bswap_16(x) \
     ((unsigned short int) ((((x) >> 8) & 0xff) | (((x) & 0xff) << 8)))

#define show(x) std::cout << #x << " : " << x  << std::endl


class weight{
public:
	long weight_id;
	bool is_fixed;
	double initial_value;
	long num_rows;
	long num_cols;
	double ** values;
};

class variable{
public:
	long matrix_id;
	long num_rows;
	long num_cols;
	vector<bool> is_evid;
	vector<double> init_value;
	long layer;
	long start_ind;
    long var_layer_ind;
};

class hyper_edge{
public:
	long num_inputs;				//number of input centers
	vector<long> in_mat_ids;   	//input centers MatIds
	vector<long> in_center_xs;	//input centers x axis 
	vector<long> in_center_ys;	//input centers y axis
	long out_mat_id;			//output center MatId
	long out_center_x;			//output center x axis
	long out_center_y;			//output center y axis
	long factor_function;		//type of the factor function
	vector<long> weight_ids;		//Weight ids
	long layer;
	long start_ind;
};

class neural_network {
public:
    variable * variables;  // Should be sorted by matrix_id
    weight * weights;
    hyper_edge * hedges;
    // vector<variable> variables;
    // vector<weight> weights;
    // vector<hyper_edge> hedges;

    long num_vars;
    long num_weights;
    long num_hedges;

    void init(long n_var, long n_weight, long n_hedges){
        variables=new variable[n_var];
        weights=new weight[n_weight];
        hedges=new hyper_edge[n_hedges];
        num_vars=n_var;
        num_weights=n_weight;
        num_hedges=n_hedges;
    }

    long load_variables(std::string filename){ 
        std::cout << "Loading variables ...\n";

        std::ifstream file;
        file.open(filename.c_str(), std::ios::in | std::ios::binary);
        long long count = 0;
        long long a = 0;

        long mid;
        	long num_rows;
        	long num_cols;

        while (file.good()) {
        	variable v;
            if (!file.read((char *)&mid, 8)) break;
            file.read((char *)&num_rows, 8);
            file.read((char *)&num_cols, 8);
            mid = bswap_64(mid);

            num_rows = bswap_64(num_rows);
            num_cols = bswap_64(num_cols);

            bool is_evidence;
            for(int i=0; i<num_rows; i++)
            	for(int j=0; j<num_cols; j++){
            		file.read((char *)&is_evidence, 1);
            		v.is_evid.push_back(is_evidence);
            	}
            uint64_t initial_value;
            for(int i=0; i<num_rows; i++)
            	for(int j=0; j<num_cols; j++){
            		file.read((char *)&initial_value, 8);
            		initial_value = bswap_64(*(uint64_t *)&initial_value);
                    double initval = *(double *)&initial_value;
            		v.init_value.push_back(initval);
            	}
            bool temp;
            long layer;
            if (!file.read((char *)&layer, 8)) break;
            layer = bswap_64(layer);

            v.matrix_id=mid;
        	v.num_rows=num_rows;
        	v.num_cols=num_cols;
        	v.layer=layer;

            // show(v.matrix_id);
            // show(num_vars);
            // if(v.matrix_id>num_vars){
            //     show("************");
            //     exit(0);
            // }
            //std::cout << "V" << n.id << "(" << n.is_evid << ")" << " ---> " << n.init_value << std::endl;
            // variables.push_back(v);
            variables[v.matrix_id]=v;
            count++;
        }
        file.close();
        return count;

    }

    long load_weights(std::string filename){
        std::cout << "Loading weights ...\n";

        std::ifstream file;
        file.open(filename.c_str(), std::ios::in | std::ios::binary);
        long weight_id;
        bool is_fixed;
        double initval;
        uint64_t initial_value;
        long num_rows;
        long num_cols;
        long count=0;
        while (file.good()) {
          	// read fields
            if (!file.read((char *)&weight_id, 8)) break;
            file.read((char *)&num_rows, 8);
            file.read((char *)&num_cols, 8);

            file.read((char *)&is_fixed, 1);
            if(!file.read((char *)&initial_value, 8)) break;

            // convert endian
            weight_id = bswap_64(weight_id);

            initial_value = bswap_64(*(uint64_t *)&initial_value);
            initval = *(double *)&initial_value;


            num_rows = bswap_64(num_rows);
            num_cols = bswap_64(num_cols);

            // show(weight_id);
            // show(is_fixed);
            // show(initval);
            // show(num_rows);
            // show(num_cols);


            weight w;
            w.weight_id = weight_id;
            w.is_fixed = is_fixed;
            w.initial_value = initval;
            w.num_rows = num_rows;
            w.num_cols = num_cols;

            // weights.push_back(w);
            // show(w.weight_id);
            // show(num_weights);
            // if(w.weight_id>num_weights){
            //     show("************");
            //     exit(0);
            // }
            weights[w.weight_id]=w;
            count++;
        }
        file.close();
        return count;
    }

    long load_edges(std::string filename){
        std::cout << "Loading edges\n";

        std::ifstream file;
        file.open(filename.c_str(), std::ios::in | std::ios::binary);
        long long count = 0;

        long num_inputs;
        vector<long> in_mat_ids;
        vector<long> in_center_xs;
        vector<long> in_center_ys;
        long out_mat_id;
        long out_center_x;
        long out_center_y;
        long factor_function;
        vector<long> weight_ids;

        while (file.good()) {
        	hyper_edge he;
          	if(!file.read((char *)&num_inputs, 8)) break;
            num_inputs = bswap_64(num_inputs);
            // show(num_inputs);


        	long in_mat_id;
            for(int i=0; i<num_inputs; i++){
        		file.read((char *)&in_mat_id, 8);
        		in_mat_id = bswap_64(in_mat_id);
                // show(in_mat_id);
        		he.in_mat_ids.push_back(in_mat_id);
        	}
        	long in_center_x;
            for(int i=0; i<num_inputs; i++){
        		file.read((char *)&in_center_x, 8);
        		in_center_x = bswap_64(in_center_x);
                // show(in_center_x);
        		he.in_center_xs.push_back(in_center_x);
        	}

        	long in_center_y;
            for(int i=0; i<num_inputs; i++){
        		file.read((char *)&in_center_y, 8);
        		in_center_y = bswap_64(in_center_y);
                // show(in_center_y);
        		he.in_center_ys.push_back(in_center_y);
        	}


            file.read((char *)&out_mat_id, 8);
            file.read((char *)&out_center_x, 8);
            file.read((char *)&out_center_y, 8);
            file.read((char *)&factor_function, 8);
        	out_mat_id = bswap_64(out_mat_id);
            out_center_x = bswap_64(out_center_x);
            out_center_y = bswap_64(out_center_y);
            factor_function = bswap_64(factor_function);

            // show(out_mat_id);
            // show(out_center_x);
            // show(out_center_y);
            // show(factor_function);


        	long weight_id;
            for(int i=0; i<num_inputs; i++){
        		file.read((char *)&weight_id, 8);
        		weight_id = bswap_64(weight_id);
                // show(weight_id);
        		he.weight_ids.push_back(weight_id);
        	}

            he.num_inputs=num_inputs;
            he.out_mat_id=out_mat_id;
            he.out_center_x=out_center_x;
            he.out_center_y=out_center_y;
            he.factor_function=factor_function;
            hedges[count]=he;
            count++;
        }
        file.close();
        return count;   
    }

    void load(const dd::CmdParser & cmd, long _n_var, long _n_weight, long _n_hedges){
        std::cout << "Loading Neural Network .";

        std::string weight_file = cmd.weight_file->getValue();
        std::string variable_file = cmd.variable_file->getValue();
        std::string edge_file = cmd.edge_file->getValue();
        std::string fg_file = cmd.fg_file->getValue();


        std::cout << ".";

        std::string filename_edges = edge_file;
        std::string filename_variables = variable_file;
        std::string filename_weights = weight_file;
        std::string filename_meta= fg_file;
        std::cout << ".\n";

        init(_n_var,_n_weight,_n_hedges);

        long n_var = load_variables(filename_variables);
        long n_weight = load_weights(filename_weights);
        long n_hedges = load_edges(filename_edges);

        if(n_var!=_n_var){
            std::cout << "ERROR: NUMBER OF VARIABLES FROM FILE IS DIFFERENT FROM NUMBER OF VARIABLES LOADED INTO NETWORK";
            exit(0);
        }

        if(n_weight!=_n_weight){
            std::cout << "ERROR: NUMBER OF WEIGHTS FROM FILE IS DIFFERENT FROM NUMBER OF WEIGHTS LOADED INTO NETWORK";
            exit(0);
        }

        if(n_hedges!=_n_hedges){
            std::cout << "ERROR: NUMBER OF HYPER_EDGES FROM FILE IS DIFFERENT FROM NUMBER OF HYPER_EDGES LOADED INTO NETWORK";
            exit(0);
        }

        // std::sort(&neurons[0], &neurons[n_var], idsorter<Neuron>());
        // std::sort(&connections[0], &connections[n_factor], idsorter<Connection>());
        // std::sort(&weights[0], &weights[n_weight], idsorter<Weight>()); 

        // std::cout << "NVAR = " << n_var << std::endl;
        // std::cout << "NWEI = " << n_weight << std::endl;
        // std::cout << "NEDG = " << n_hedges << std::endl;
    }
};

#endif
