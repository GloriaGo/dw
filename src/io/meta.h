#ifndef META_H
#define META_H

using namespace std;

// meta data
typedef struct {
	long long num_weights;
	long long num_variables;
	long long num_edges;
	string weights_file; 
	string variables_file;
	string edges_file;
} Meta;

// Read meta data file, return Meta struct 
Meta read_meta(string meta_file)
{
	ifstream file;
	file.open(meta_file.c_str());
	string buf;
	Meta meta;
	getline(file, buf, ',');
	meta.num_weights = atoll(buf.c_str());
	getline(file, buf, ',');
	meta.num_variables = atoll(buf.c_str());
	getline(file, buf, ',');
	meta.num_edges = atoll(buf.c_str());
	getline(file, meta.weights_file, ',');
	getline(file, meta.variables_file, ',');
	getline(file, meta.edges_file, ',');
	file.close();
	return meta;
}


#endif