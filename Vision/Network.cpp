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

void Network::NormalInitialization()
{
	std::cout << "Initialize weights and biases" << std::endl;

	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, 1.0);
	auto normal = [&](float) {return distribution(generator); };


	for(int i = 1; i < Layers.size(); i++) 
	{
		Biases.push_back(Eigen::VectorXf::NullaryExpr(Layers[i].Size(), normal));
		Weights.push_back(Eigen::MatrixXf::NullaryExpr(Layers[i].Size(),Layers[i - 1].Size(), normal));
	
	}

	std::cout << "Initialization finished" << std::endl;
}

Eigen::VectorXf Network::Forward(Eigen::VectorXf A)
{

	Eigen::VectorXf Z;


	for (int iLayers = 0; iLayers < Layers.size(); iLayers++) 
	{
		Z = Weights[iLayers] * A + Biases[iLayers];
		A = Function::ActivationFunc(Activation::sigmoid, Z);
	}
	return A;
}
