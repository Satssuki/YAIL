#include "Network.h"



Network::Network()
{
}

Network::Network(std::vector<Layer*> layers)
{
	Layers = layers;
}


Network::~Network()
{
}

void Network::Add(Layer* layer)
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

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n + 1;

	std::vector<T> vec(first, last);
	return vec;
}

void Network::Train()
{
	for (int e = 0; e < Epoch; e++)
	{
		std::vector<std::tuple<std::vector<cv::Mat>, std::vector<int>>> batches;
		int trainDataSize = std::get<0>(TrainData).size();
		int totalBatches = ceil(trainDataSize / (float)BatchSize);
		for (int b = 0; b < totalBatches; b++)
		{
			std::vector<cv::Mat> batchImages;
			std::vector<int> batchLabels;
			int i = b * BatchSize;

			if (b == totalBatches - 1)
			{
				batchImages = slice(std::get<0>(TrainData), i, (trainDataSize - 1));
				batchLabels = slice(std::get<1>(TrainData), i, (trainDataSize - 1));
			}
			else
			{
				batchImages = slice(std::get<0>(TrainData), i, i + BatchSize - 1);
				batchLabels = slice(std::get<1>(TrainData), i, i + BatchSize - 1);
			}

			UpdateBatch({ batchImages, batchLabels });
		}

		std::cout << "Epoch " << e << " : " + std::to_string(Evaluate()) << " / " << std::to_string(std::get<0>(TestData).size()) << std::endl;
	}
	std::cout << "Training completed" << std::endl;
}

std::vector<float> Network::Predict(cv::Mat image)
{
	return std::vector<float>();
}

void Network::Summary()
{
	std::cout << "Layer (type) | Output Shape | Param #" << std::endl;
	std::cout << Layers[0]->ToString() + " | " + std::to_string(Layers[0]->Size()) + " | 0" << std::endl;
	for (int i = 1; i < Layers.size(); i++)
	{
		std::cout << Layers[i]->ToString() + " | " + std::to_string(Layers[i]->Size()) + " | " + std::to_string(Layers[i - 1]->Size() * Layers[i]->Size() + Layers[i]->Size()) << std::endl;
	}
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
		Biases.push_back(Eigen::VectorXf::NullaryExpr(Layers[i]->Size(), normal));
		for (int iw = 0; iw < Layers[i]->Size(); iw++)
		{
			Weights[i - 1].push_back(Eigen::VectorXf::NullaryExpr(Layers[i - 1]->Size(), normal));
		}	
	}

	std::cout << "Initialization finished" << std::endl;
}

float Network::Evaluate()
{
	return 0.0f;
}

void Network::UpdateBatch(std::tuple<std::vector<cv::Mat>, std::vector<int>> batch)
{
}

Eigen::VectorXf Network::Forward(Eigen::VectorXf input)
{
	return Eigen::VectorXf();
}
