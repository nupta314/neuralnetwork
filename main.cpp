#include <string>

#include "neural.hpp"

using namespace std;

/** COMPLETE
Except tilting and my own handwriting
**/

int main() {
	const int input_nodes = 784,hidden_nodes = 100,output_nodes = 10;
	const fract learning_rate=0.1;
	//0.3 -> 94.81
	//0.25-> 94.83
	//0.2->95.12
	//0.18->95.2
	//0.01 -> 97.47 (22)
	//0.01 & 200(hno) - 97.7
	cout<<"Initializing neural network\n";
	neuralNetwork brain(input_nodes,hidden_nodes,output_nodes,learning_rate);
	for(int i = 1; i<=10 ; ++i) {
		cout<<"Iteration #"<<i<<'\n';
		train(brain,"data/mnist_train.csv");
	}
	test(brain,"data/mnist_test.csv");
	for(auto i=0; i<10; ++i) {
		dmat a = Num2Dmat(i);
		dmat b = brain.bquery(a,true);
		vector<int> c = Dmat2Vec(b);
		displayImg(c,"it is supposed to be digit "+to_string(i));
	}
	return 0;
}
