#pragma once

#include "AllIncludes.h"
#include <opencv2/core/eigen.hpp>

#include "Optimizer.h"
#include "Loss.h"

#include "Dense.h"
#include "Input.h"

class Network
{
public:
	Network();
	Network(std::vector<Layer*> layers);
	~Network();

	void Add(Layer* layer);
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
	int Evaluate();
	void UpdateBatch(std::tuple < std::vector<cv::Mat>, std::vector<int>> batch);
	std::tuple < std::vector < Eigen::MatrixXf>, std::vector<Eigen::VectorXf> > BackPropagation(Eigen::VectorXf image, int label);
	Eigen::VectorXf Forward(Eigen::VectorXf input);
	Eigen::VectorXf ConvertLabel2LastLayer(int label);

	std::vector<Layer*> Layers;
	std::vector<Eigen::VectorXf> Biases;
	std::vector<Eigen::MatrixXf> Weights;

	std::tuple<std::vector<cv::Mat>, std::vector<int>> TrainData;
	std::tuple<std::vector<cv::Mat>, std::vector<int>> TestData;

	Eigen::IOFormat CleanFmt;

	Optimizer _Optimizer;
	Loss _Loss;
	int Epoch;
	int BatchSize;
	float LearningRate;
};

