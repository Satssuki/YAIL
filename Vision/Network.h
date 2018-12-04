#pragma once

#include "AllIncludes.h"

#include "Optimizer.h"
#include "Loss.h"

#include "Dense.h"
#include "Input.h"

class Network
{
public:
	Network();
	Network(std::vector<Layer> layers);
	~Network();

	void Add(Layer layer);
	void Compile(std::string optimizer = "sgd", std::string loss = "mse");
	void Compile(Optimizer optimizer = stochastic_gradient_descent, Loss loss = mean_squared_error);
	void Fit(std::tuple < std::vector<cv::Mat>, std::vector<int>> train, std::tuple < std::vector<cv::Mat>, std::vector<int>> test);
	void Hyperparameter(int epoch, int batchSize, float learningRate);
	void SaveWeights(std::string filename);
	void LoadWeights(std::string filename);
	void Train();
	std::vector<float> Predict(cv::Mat image);
	void Summary();
	void Plot();	

private:
	void NormalInitialization();
	Eigen::VectorXf Forward(Eigen::VectorXf input);

	std::vector<Layer> Layers;
	std::vector<Eigen::VectorXf> Biases;
	std::vector< std::vector<Eigen::VectorXf>> Weights;
};

