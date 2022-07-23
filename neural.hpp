#ifndef NEURAL_HPP_INCLUDED
#define NEURAL_HPP_INCLUDED

#include <iostream>
#include <vector>
#include <ranges>

#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>

#include <rapidcsv/rapidcsv.h>
#include <random>

typedef double fract;
typedef dlib::matrix<fract> dmat;

class neuralNetwork {
	int inodes,hnodes,onodes;
	fract lr;
	dmat wih,who;
  public:
	neuralNetwork(int inputnodes,int hiddennodes,int outputnodes,fract learningrate);
	void train(dmat inputs,dmat targets);
	dmat query(dmat inputs);
	dmat bquery(dmat targets,bool myVersion=false);
};

dmat Num2Dmat(int n);

std::vector<int> Dmat2Vec(dmat inp);

void displayImg(std::vector<int> img,std::string title="\0");

void train(neuralNetwork& demo,std::string filename="data/mnist_train.csv",bool print = false);

fract test(neuralNetwork& demo,std::string filename="data/mnist_test.csv");

#endif // NEURAL_HPP_INCLUDED
