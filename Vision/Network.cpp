#include "Network.h"



Network::Network()
{
}

Network::Network(std::vector<Layer> layers)
{
	Layers = layers;
}


Network::~Network()
{
}

void Network::Add(Layer layer)
{
	Layers.push_back(layer);
}

void Network::Compile(std::string optimizer, std::string loss)
{
	NormalInitialization();
}

void Network::Compile(Optimizer optimizer, Loss loss)
{
	NormalInitialization();
}

void Network::Fit(std::tuple<std::vector<cv::Mat>, std::vector<int>> train, std::tuple<std::vector<cv::Mat>, std::vector<int>> test)
{
	TrainData = train;
	TestData = test;
}

void Network::Hyperparameter(int epoch, int batchSize, float learningRate)
{
	Epoch = epoch;
	BatchSize = batchSize;
	LearningRate = learningRate;
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

void Network::NormalInitialization()
{
	std::cout << "Initialize weights and biases" << std::endl;

	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, 1.0);
	auto normal = [&](float) {return distribution(generator); };

	Weights.resize(Layers.size() - 1);

	for(int i = 1; i < Layers.size(); i++) 
	{
		Biases.push_back(Eigen::VectorXf::NullaryExpr(Layers[i].Size(), normal));
		for (int iw = 0; iw < Layers[i].Size(); iw++)
		{
			Weights[i - 1].push_back(Eigen::VectorXf::NullaryExpr(Layers[i - 1].Size(), normal));
		}	
	}

	std::cout << "Initialization finished" << std::endl;
}

Eigen::VectorXf Network::Forward(Eigen::VectorXf input)
{
	return Eigen::VectorXf();
}
