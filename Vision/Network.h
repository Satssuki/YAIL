#pragma once

#include "AllIncludes.h"
#include <opencv2/core/eigen.hpp>

#include "Optimizer.h"
#include "Loss.h"

#include "Dense.h"
#include "Input.h"
#include "MaxPool2D.h"
#include "Conv2D.h"

class Network
{
public:
	Network();
	Network(std::vector<Layer*> layers);
	~Network();

	void Add(Layer* layer);
	void Compile(Optimizer optimizer = stochastic_gradient_descent, Loss loss = mean_squared_error);
	void Fit(std::tuple < std::vector<cv::Mat>, std::vector<int>> train, std::tuple < std::vector<cv::Mat>, std::vector<int>> test);
	void Hyperparameter(int epoch, int batchSize, float learningRate);
	void SaveWeights(std::string filename);
	void LoadWeights(std::string filename);
	void Train();
	int Predict(cv::Mat image);
	void Summary();

private:
	void NormalInitialization();
	std::tuple <int, float> Evaluate();
	void UpdateBatch(std::tuple < std::vector<cv::Mat>, std::vector<int>> batch);
	std::tuple < std::vector < Eigen::MatrixXf>, std::vector<Eigen::VectorXf> > BackPropagation(Eigen::VectorXf image, int label);
	std::tuple < std::vector < Eigen::MatrixXf>, std::vector<Eigen::VectorXf> > BackPropagation(Eigen::MatrixXf image, int label);

	Eigen::VectorXf Forward(Eigen::VectorXf input);
	Eigen::VectorXf ForwardCNN(Eigen::MatrixXf input);

	Eigen::VectorXf VectorizeLabel(int label);
	Eigen::VectorXf VectorizeImage(cv::Mat imageCV);
	Eigen::MatrixXf MatrixImage(cv::Mat imageCV);

	/******* Test *******/
	std::vector<Eigen::MatrixXf> Conv1;
	std::vector<Eigen::MatrixXf> Conv2;
	/**************/
	
	std::vector<Layer*> Layers;
	std::vector<Eigen::VectorXf> Biases;
	std::vector<Eigen::MatrixXf> Weights;

	std::tuple<std::vector<cv::Mat>, std::vector<int>> TrainData;
	std::tuple<std::vector<cv::Mat>, std::vector<int>> TestData;

	Optimizer _Optimizer;
	Loss _Loss;
	int Epoch;
	int BatchSize;
	float LearningRate;

	// other
	Eigen::IOFormat CleanFmt;
};

