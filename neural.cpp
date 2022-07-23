#include "neural.hpp"

typedef unsigned char pix;
const pix height=28,width=28,Max=255,zoom=25;
dlib::interpolate_nearest_neighbor inn;

dmat activation_function(dmat inp) {
	dmat ret = 1 / ( 1 + exp( -1 * inp ));
	return ret;
}

dmat inverse_activation_function(dmat tar) {
	dmat ret = log( tar )- log( 1 - tar );
	return tar;
}

neuralNetwork::neuralNetwork(int inputnodes,int hiddennodes,int outputnodes,fract learningrate) {
	inodes=inputnodes;
	onodes=outputnodes;
	hnodes=hiddennodes;
	lr=learningrate;
	std::random_device r;
	std::mt19937_64 e(r());
	std::normal_distribution<> ih(0,pow(inodes,-0.5)),ho(0,pow(hnodes,-0.5));
	wih = dmat(inodes,hnodes);
	who = dmat(hnodes,onodes);
	for(auto i:std::views::iota(0,inodes)) {
		for(auto j:std::views::iota(0,hnodes)) {
			wih(i,j)=ih(e);
		}
	}
	for(auto i:std::views::iota(0,hnodes)) {
		for(auto j:std::views::iota(0,onodes)) {
			who(i,j)=ho(e);
		}
	}
}

void neuralNetwork::train(dmat inputs_list,dmat targets_list) {
	dmat inputs=trans(inputs_list);
	dmat targets=trans(targets_list);

	dmat hidden_inputs = inputs * wih;
	dmat hidden_outputs = activation_function(hidden_inputs);

	dmat final_inputs = hidden_outputs * who;
	dmat final_outputs = activation_function(final_inputs);

	dmat output_errors = targets - final_outputs;
	dmat hidden_errors = output_errors * trans(who);

	//for 1/(1 + e^-x) activation function
	who += lr * trans(hidden_outputs) * pointwise_multiply(pointwise_multiply(output_errors,final_outputs),(1 - final_outputs));
	wih += lr * trans(inputs) * pointwise_multiply(pointwise_multiply(hidden_errors,hidden_outputs),(1 - hidden_outputs));
}

dmat neuralNetwork::query(dmat inputs_list) {
	dmat inputs = trans(inputs_list);

	dmat hidden_inputs = inputs * wih;
	dmat hidden_outputs = activation_function(hidden_inputs);

	dmat final_inputs = hidden_outputs * who;
	dmat final_outputs = activation_function(final_inputs);

	return final_outputs;
}

dmat neuralNetwork::bquery(dmat targets_list,bool myVersion) {
	dmat final_outputs = trans(targets_list);
	dmat final_inputs = inverse_activation_function(final_outputs);
	dmat hidden_outputs = final_inputs * trans(who);

	if(!(myVersion)) {
		hidden_outputs -= dlib::min(hidden_outputs);
		hidden_outputs /= dlib::max(hidden_outputs);
		hidden_outputs *=0.98;
		hidden_outputs +=0.01;
	}

	dmat hidden_inputs = inverse_activation_function(hidden_outputs);
	dmat inputs = hidden_inputs * trans(wih);

	inputs -= dlib::min(inputs);
	inputs /= dlib::max(inputs);
	inputs *= 0.98;
	inputs += 0.01;

	return inputs;
}

dmat Num2Dmat(int n) {
	dmat ret(10,1);
	ret = 0.01;
	ret(n)=0.99;
	return ret;
}

std::vector<int> Dmat2Vec(dmat inp) {
	std::vector<int> ret;
	for(auto i:std::views::iota(0,inp.size())) {
		ret.push_back(inp(i)*Max);
	}
	return ret;
}

void displayImg(std::vector<int> img,std::string title) { //vector<int> of 28*28 (784) elements
	dlib::array2d<dlib::rgb_pixel> a(height,width),b(height*zoom,width*zoom);
	for(pix i=0; i<height; ++i) {
		for(pix j=0; j<width; ++j) {
			a[i][j]=dlib::rgb_pixel{(pix)(Max-img[i*height+j]),(pix)(Max-img[i*height+j]),(pix)(Max-img[i*height+j])};
		}
	}
	dlib::resize_image(a,b,inn);
	dlib::image_window my_window(b,title);
	my_window.wait_until_closed();
}

void train(neuralNetwork& demo,std::string filename,bool print) {
	if(print)std::cout<<"Loading train database\n";
	rapidcsv::Document doc(filename,rapidcsv::LabelParams(-1,-1));
	if(print)std::cout<<"Training Started\n";
	int trainSize = doc.GetRowCount();
	for(auto i:std::views::iota(0,trainSize)) {
		std::vector<double> row = doc.GetRow<double>(i);
		dmat targets = dlib::zeros_matrix<double>(10,1);
		targets += 0.01;
		targets(row[0])=0.99;
		row.erase(row.begin());
		dmat inputs(row.size(),1);
		for(long long unsigned int j=0; j<row.size(); ++j) {
			inputs(j,0)=row[j]*0.99/Max + 0.01;
		}
		demo.train(inputs,targets);
		if(print) {
			if(!(i%(trainSize/10))) {
				std::cout<<i<<"/"<<trainSize<<'\n';
			}
		}
	}
}

fract test(neuralNetwork& demo,std::string filename) {
	fract rc=0;
	std::cout<<"Loading test database\n";
	rapidcsv::Document doc(filename,rapidcsv::LabelParams(-1,-1));
	std::cout<<"Testing started\n";
	int testSize=doc.GetRowCount();
	for(auto i:std::views::iota(0,testSize)) {
		std::vector<int> row = doc.GetRow<int>(i);
		dmat inputs(row.size()-1,1);
		for(long long unsigned int j=0; j<row.size()-1; ++j) {
			inputs(j,0)=row[j+1]*0.99/Max + 0.01;
		}
		dmat output = demo.query(inputs);
		rc+=(row[0]==dlib::index_of_max(output))?1:0;
	}
	rc = rc/testSize*100;
	std::cout<<"The result is "<<rc<<"%\n";
	return rc;
}
