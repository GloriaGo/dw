#ifndef _NEURAL_MODEL_H
#define _NEURAL_MODEL_H

#include "io/cmd_parser.h"
#include <map>
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

#define show(x)     std::cout.precision(11); std::cout << #x << " : " << x  << std::endl


class weight{
public:
	long weight_id;
	bool is_fixed;
	double initial_value;
	long num_rows;
	long num_cols;
	double * values;
	double bias;

	long mem_size(){
		return sizeof(long)*3+ sizeof(bool)+ sizeof(double)*(2+num_rows*num_cols);
	}
};

class image{
public:
	image(){
		fid=0;
	}
	long image_id;
	long fid;
	long num_rows;
	long num_cols;
	double * values[3];
	long label;
};

class variable{
public:
	long image_id;
	long fid;
	long matrix_id;
	long num_rows;
	long num_cols;
	bool * is_evid;
	double * init_value;
	long layer;
	long start_ind;

	bool operator < (const variable & x)const{
		if(image_id==x.image_id){
			if(fid==x.fid)
				return layer<x.layer;
			return fid<x.fid;
		}
		return image_id<x.image_id;
	}

	long size(){
		return sizeof(long)*7+ sizeof(is_evid)+ sizeof(init_value);
	}
};

class edge{
public:
	long in_mat_id;    //input center MatIds
	long in_center_x;  //input center x axis 
	long in_center_y;  //input center y axis
	long weight_id;    //Weight id
};

class hyper_edge{
public:
	long num_inputs;               //number of input centers
	edge * start_ind;              //input edges
	long out_mat_id;               //output center MatId
	long out_center_x;             //output center x axis
	long out_center_y;             //output center y axis
	long factor_function;          //type of the factor function
	long layer;
	bool operator < (const hyper_edge & x)const{
		if(layer==x.layer){
			if(out_mat_id==x.out_mat_id){
				if(out_center_x==x.out_center_x)
					return out_center_y<x.out_center_y;
				return out_center_x<x.out_center_x;
			}
			return out_mat_id<x.out_mat_id;
		}
		return layer<x.layer;
	}
	long size(){
		return sizeof(long)*6+sizeof(8)+sizeof(edge)*num_inputs;
	}
};

long * map_id;
class neural_network {
public:
	variable * variables;  // Should be sorted by matrix_id
	weight * weights;
	hyper_edge * hedges;
	edge * hedge_inputs;
	image * images;  // Should be sorted by matrix_id


	long num_vars;
	long num_weights;
	long num_hedges;
	long num_edges;
	long num_images;

	void init(long n_var, long n_weight, long n_hedges){
		variables=new variable[n_var];
		weights=new weight[n_weight];
		hedges=new hyper_edge[n_hedges];
		images=new image[n_var];
		map_id=new long[n_var];
		num_vars=0;
		num_weights=n_weight;
		num_hedges=n_hedges;
		num_images=0;
	}

	long load_variables(std::string filename){ 
		std::cout << "Loading variables ...\n";

		std::ifstream file;
		file.open(filename.c_str(), std::ios::in | std::ios::binary);
		long count = 0;

		long image_id;
		long fid;
		long mid;
		long num_rows;
		long num_cols;
		while (file.good()) {
			variable v;
			image img;
			if (!file.read((char *)&image_id, 8)) break;
			file.read((char *)&fid, 8);
			file.read((char *)&mid, 8);
			file.read((char *)&num_rows, 8);
			file.read((char *)&num_cols, 8);

			image_id = bswap_64(image_id);
			fid = bswap_64(fid);
			mid = bswap_64(mid);
			num_rows = bswap_64(num_rows);
			num_cols = bswap_64(num_cols);

			v.is_evid=new bool [num_rows*num_cols];
			v.init_value= new double [num_rows*num_cols];
			if(image_id!=-1){
				bool is_evidence;
				for(int i=0; i<num_rows; i++)
					for(int j=0; j<num_cols; j++){
						file.read((char *)&is_evidence, 1);
						v.is_evid[i*num_cols+j]=is_evidence;
					}
				uint64_t initial_value;
				for(int i=0; i<num_rows; i++)
					for(int j=0; j<num_cols; j++){
						file.read((char *)&initial_value, 8);
						initial_value = bswap_64(*(uint64_t *)&initial_value);
						double initval = double(*(double *)&initial_value);
						v.init_value[i*num_cols+j]=initval;
					}
			}
			long layer;
			if (!file.read((char *)&layer, 8)) break;
			layer = bswap_64(layer);

			v.image_id=image_id;
			v.fid=fid;
			v.matrix_id=mid;
			v.num_rows=num_rows;
			v.num_cols=num_cols;
			v.layer=layer;
			// show(image_id);
			// show(fid);
			// show(mid);
			// show(num_rows);
			// show(num_cols);
			// show(layer);

			if(v.image_id==-1){
				map_id[v.matrix_id]=num_vars;
				variables[num_vars]=v;
				num_vars++;
			}
			else{
				images[num_images].image_id=num_images;
				images[num_images].fid++;
				images[num_images].num_rows=num_rows;
				images[num_images].num_cols=num_cols;
				images[num_images].values[fid]=v.init_value;
				images[num_images].label=layer;
				num_images++;
			}
			count++;
		}
	    // sort(variables, variables + num_vars);
		// for(int i=0; i<num_vars; i++){
		//     cout << variables[i].image_id << " " << variables[i].fid << " " << variables[i].matrix_id << " " 
		//         << variables[i].num_rows << " " << variables[i].num_cols << endl;
		// }
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
			show(initial_value);
			initval = double(*(double *)&initial_value);


			num_rows = bswap_64(num_rows);
			num_cols = bswap_64(num_cols);


			weight w;
			w.weight_id = weight_id;
			w.is_fixed = is_fixed;
			w.initial_value = initval;
			w.num_rows = num_rows;
			w.num_cols = num_cols;

			weights[w.weight_id]=w;
			count++;
		}
		file.close();
		return count;
	}

	long load_edges(std::string filename){
		std::cout << "Loading edges ...\n";

		std::ifstream file;
		file.open(filename.c_str(), std::ios::in | std::ios::binary);
		long count = 0;

		edge * temp_edges=new edge [num_hedges*1000];

		long edge_ind=0;

		long num_inputs;
		long out_mat_id;
		long out_center_x;
		long out_center_y;
		long factor_function;
		vector<long> weight_ids;

		bool myprint=1;
		int last_layer=-1;

		while (file.good()) {
			hyper_edge he;
			if(!file.read((char *)&num_inputs, 8)) break;
			num_inputs = bswap_64(num_inputs);
			// show(num_inputs);

			he.start_ind=&temp_edges[edge_ind];
			long in_mat_id;
			for(int i=0; i<num_inputs; i++){
				file.read((char *)&in_mat_id, 8);
				in_mat_id = bswap_64(in_mat_id);
				// show(in_mat_id);
				temp_edges[edge_ind+i].in_mat_id=map_id[in_mat_id];
			}
			long in_center_x;
			for(int i=0; i<num_inputs; i++){
				file.read((char *)&in_center_x, 8);
				in_center_x = bswap_64(in_center_x);
				// show(in_center_x);
				temp_edges[edge_ind+i].in_center_x=in_center_x;
			}

			long in_center_y;
			for(int i=0; i<num_inputs; i++){
				file.read((char *)&in_center_y, 8);
				in_center_y = bswap_64(in_center_y);
				// show(in_center_y);
				temp_edges[edge_ind+i].in_center_y=in_center_y;
			}


			file.read((char *)&out_mat_id, 8);
			file.read((char *)&out_center_x, 8);
			file.read((char *)&out_center_y, 8);
			file.read((char *)&factor_function, 8);
			out_mat_id = bswap_64(out_mat_id);
			out_center_x = bswap_64(out_center_x);
			out_center_y = bswap_64(out_center_y);
			factor_function = bswap_64(factor_function);




			long weight_id;
			for(int i=0; i<num_inputs; i++){
				file.read((char *)&weight_id, 8);
				weight_id = bswap_64(weight_id);
				temp_edges[edge_ind+i].weight_id=weight_id;
			}

			he.num_inputs=num_inputs;
			he.out_mat_id=map_id[out_mat_id];
			he.out_center_x=out_center_x;
			he.out_center_y=out_center_y;
			he.factor_function=factor_function;
			// show(num_inputs);
			// show(out_mat_id);
			// show(out_center_x);
			// show(out_center_y);
			// show(factor_function);

			hedges[count]=he;
			count++;
			edge_ind+=num_inputs;
		}
		file.close();
		sort(hedges, hedges + num_hedges);

		hedge_inputs= new edge [edge_ind];
		edge_ind=0;
		for(int i=0; i<num_hedges; i++){
			// show(edge_ind);
			for(int j=0; j<hedges[i].num_inputs; j++){
				hedge_inputs[edge_ind+j].in_mat_id=hedges[i].start_ind[j].in_mat_id;
				hedge_inputs[edge_ind+j].in_center_x=hedges[i].start_ind[j].in_center_x;
				hedge_inputs[edge_ind+j].in_center_y=hedges[i].start_ind[j].in_center_y;
				hedge_inputs[edge_ind+j].weight_id=hedges[i].start_ind[j].weight_id;
			}
			// show(edge_ind);
			hedges[i].start_ind=&hedge_inputs[edge_ind];
			edge_ind+=hedges[i].num_inputs;
		}

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

		std::cout << "Loading Done :-)" << endl;

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
