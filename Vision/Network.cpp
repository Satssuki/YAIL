#include "Network.h"



Network::Network()
{
	CleanFmt = Eigen::IOFormat(4, 0, ", ", ";\n", "[", "]", "[", "]");
}

Network::Network(std::vector<Layer*> layers)
{
	Layers = layers;
	CleanFmt = Eigen::IOFormat(4, 0, ", ", ";\n", "[", "]", "[", "]");
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
	
}

void Network::Compile(Optimizer optimizer, Loss loss)
{
	_Optimizer = optimizer;
	_Loss = loss;

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
		int trainDataSize = std::get<0>(TrainData).size();
		int totalBatches = ceil(trainDataSize / (float)BatchSize);
		for (int b = 0; b < totalBatches; b++)
		{
			clock_t startTime = clock();

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

			if (b % 100 == 0) { std::cout << "\rBatch " << b << " / " << totalBatches; }
		}
		std::cout << "\r                     ";
		std::cout << "\rEpoch " << e << " : " + std::to_string(Evaluate()) << " / " << std::to_string(std::get<0>(TestData).size()) << std::endl;
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
	int total = 0;
	for (int i = 1; i < Layers.size(); i++)
	{
		total += Layers[i - 1]->Size() * Layers[i]->Size() + Layers[i]->Size();
		std::cout << Layers[i]->ToString() + " | " + std::to_string(Layers[i]->Size()) + " | " + std::to_string(Layers[i - 1]->Size() * Layers[i]->Size() + Layers[i]->Size()) << std::endl;
	}
	std::cout <<  "Total |  | " << total << std::endl;
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
		Biases.push_back(Eigen::VectorXf::NullaryExpr(Layers[i]->Size(), normal));		
		Weights.push_back(Eigen::MatrixXf::NullaryExpr(Layers[i]->Size(),Layers[i - 1]->Size(), normal));
	}

	std::cout << "Initialization finished" << std::endl;
}

int Network::Evaluate()
{
	// this is a test put two loops into one method
	std::vector<int> predictions;
	for (int i = 0; i < std::get<0>(TestData).size(); i++)
	{
		// Todo call function to convert opencv mat to eigen vec
		cv::Mat imageCV = std::get<0>(TestData)[i];
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(reinterpret_cast<float*>(imageCV.data), imageCV.rows, imageCV.cols * imageCV.channels());
		Eigen::Map<Eigen::RowVectorXf> image(mat.data(), mat.size());
		Eigen::VectorXf L = Forward(image);
		float max = L(0);
		int maxIndex = 0;
		for (int v = 1; v < L.rows(); v++)
		{
			float n = L(v);
			if (max < n)
			{
				max = n;
				maxIndex = v;
			}
		}
		predictions.push_back(maxIndex);
	}

	int goodResult = 0;
	for (int i = 0; i < std::get<0>(TestData).size(); i++)
	{
		if (std::get<1>(TestData)[i] == predictions[i])
		{
			goodResult++;
		}
	}

	return goodResult;
}

void Network::UpdateBatch(std::tuple<std::vector<cv::Mat>, std::vector<int>> batch)
{
	int currentBatchSize = std::get<0>(batch).size();

	std::vector<Eigen::MatrixXf> sumWeightsError;
	std::vector<Eigen::VectorXf> sumBiasesError;
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		sumWeightsError.push_back(Eigen::MatrixXf::Zero(Weights[i].rows(), Weights[i].cols())); // Hint: maybe replace order?
		sumBiasesError.push_back(Eigen::VectorXf::Zero(Biases[i].rows()));
	}

	for (int i = 0; i < currentBatchSize; i++)
	{
		cv::Mat imageCV = std::get<0>(batch)[i];

		// Todo call function to convert opencv mat to eigen vec
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(reinterpret_cast<float*>(imageCV.data), imageCV.rows, imageCV.cols * imageCV.channels());
		Eigen::Map<Eigen::RowVectorXf> image(mat.data(), mat.size());

		int label = std::get<1>(batch)[i];
		auto errorWeightsBiases = BackPropagation(image, label);

		for (int iL = 0; iL < Layers.size() - 1; iL++)
		{
			sumWeightsError[iL].array() += std::get<0>(errorWeightsBiases)[iL].array();
			sumBiasesError[iL].array() += std::get<1>(errorWeightsBiases)[iL].array();
		}	
	}
	
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		Weights[i].array() = Weights[i].array() - (LearningRate / (float)currentBatchSize) * sumWeightsError[i].array();
		Biases[i].array() = Biases[i].array() - (LearningRate / (float)currentBatchSize) * sumBiasesError[i].array();
	}
}

std::tuple<std::vector<Eigen::MatrixXf>, std::vector<Eigen::VectorXf>> Network::BackPropagation(Eigen::VectorXf image, int label)
{
	std::vector< Eigen::VectorXf> activations;
	activations.push_back(image);

	std::vector< Eigen::VectorXf> zs;
	Eigen::VectorXf a = image;
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		Eigen::VectorXf z = Weights[i] * a + Biases[i];
		a = Function::ActivationFunction(Layers[i + 1]->_Activation, z);

		zs.push_back(z);
		activations.push_back(a);
	}

	std::vector< Eigen::MatrixXf> deltaWeights;
	std::vector< Eigen::VectorXf> deltaBiases;

	Eigen::VectorXf costPrime = Function::ErrorFunction(_Loss, ConvertLabel2LastLayer(label), activations.back());
	Eigen::VectorXf sigmoidPrime = Function::ActivationFunctionPrime(Layers.back()->_Activation, zs.back());
	Eigen::VectorXf delta = costPrime.array() * sigmoidPrime.array();

	deltaBiases.push_back(delta);
	deltaWeights.push_back(delta * activations[activations.size() - 2].transpose());

	for (int i = 2; i < Layers.size(); i++)
	{
		Eigen::VectorXf sp = Function::ActivationFunctionPrime(Layers[Layers.size() - i]->_Activation, zs[zs.size() - i]);
		delta = (Weights[Weights.size() - i + 1].transpose() * delta).array() * sp.array();

		deltaBiases.insert(deltaBiases.begin(), delta);
		deltaWeights.insert(deltaWeights.begin(), delta * activations[activations.size() - i - 1].transpose());
	}

	return {deltaWeights, deltaBiases};
}

Eigen::VectorXf Network::Forward(Eigen::VectorXf input)
{	
	Eigen::VectorXf a = input;
	for (int iL = 0; iL < Layers.size() - 1; iL++) 
	{
		Eigen::VectorXf z = Weights[iL] * a + Biases[iL];
		a = Function::ActivationFunction(Layers[iL + 1]->_Activation, z);
	}
	
	return a;
}

Eigen::VectorXf Network::ConvertLabel2LastLayer(int label)
{
	Eigen::VectorXf lastLayer = Eigen::VectorXf::Zero(Layers.back()->Size());
	lastLayer(label) = 1;

	return lastLayer;
}
