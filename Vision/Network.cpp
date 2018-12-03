#include "Network.h"



Network::Network()
{
}

Network::Network(std::vector<Layer*> layers)
{
}


Network::~Network()
{
}

void Network::Add(Layer* layer)
{
}

void Network::Compile(std::string optimizer, std::string loss)
{
}

void Network::Fit(std::tuple<std::vector<cv::Mat>, std::vector<int>> train, std::tuple<std::vector<cv::Mat>, std::vector<int>> test)
{
}

void Network::Hyperparameter(int epoch, int batchSize, float learningRate)
{
}

void Network::SaveWeights(std::string filename)
{
}

void Network::LoadWeights(std::string filename)
{
}

void Network::Train()
{
}

std::vector<float> Network::Predict(cv::Mat image)
{
	return std::vector<float>();
}

void Network::Summary()
{
}

void Network::Plot()
{
}
