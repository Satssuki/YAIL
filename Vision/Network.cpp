#include "Network.h"


template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n + 1;

	std::vector<T> vec(first, last);
	return vec;
}

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
	std::ofstream of;
	of.open(filename, std::ios::binary | std::ios::out);
	
	// save weights first
	for (int i = 0; i < Weights.size();  i++)
	{
		EigenSerializer::SaveMatrix(of, Weights[i]);
	}

	// save biases
	for (int i = 0; i < Biases.size(); i++)
	{
		EigenSerializer::SaveVector(of, Biases[i]);
	}

	of.close();
}

void Network::LoadWeights(std::string filename)
{
	std::ifstream inf;
	inf.open(filename, std::ios::binary | std::ios::in);

	// load weights first
	for (int i = 0; i < Weights.size(); i++)
	{
		EigenSerializer::LoadMatrix(inf, Weights[i]);
	}

	// load biases
	for (int i = 0; i < Biases.size(); i++)
	{
		EigenSerializer::LoadVector(inf, Biases[i]);
	}

	inf.close();
}

void Network::Train()
{
	std::clock_t start = std::clock();

	for (int e = 0; e < Epoch; e++)
	{
		//// to delete
		//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		//shuffle(std::get<0>(TrainData).begin(), std::get<0>(TrainData).end(), std::default_random_engine(seed));
		//shuffle(std::get<1>(TrainData).begin(), std::get<1>(TrainData).end(), std::default_random_engine(seed));

		// maybe should shuffle training data each time lol
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
		auto evaluation = Evaluate();
		std::cout << "\r                     ";
		std::cout << "\rEpoch " << e << ". Acc : " + std::to_string(std::get<0>(evaluation)) << " / " << std::get<0>(TestData).size() << " Loss: " << std::setprecision(2) << std::get<1>(evaluation) << std::endl;
	}

	clock_t end = clock();
	clock_t ticks = end - start;
	double secs = ticks / (double)CLOCKS_PER_SEC;
	float min =	fmod(secs / 60, 60);
	
	std::cout << "Training completed. Took " << min << " min" << std::endl;
}

int Network::Predict(cv::Mat image)
{
	Eigen::VectorXf imageEigen = cv2eigen::Convert(image);
	Eigen::VectorXf L = Forward(imageEigen);
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

	return maxIndex;
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
	auto normalBias = [&](float) {return distribution(generator); };

	for(int i = 1; i < Layers.size(); i++) 
	{
		Biases.push_back(Eigen::VectorXf::NullaryExpr(Layers[i]->Size(), normalBias));	

		auto normalWeight = [&](float) {return distribution(generator)/sqrt(Layers[i - 1]->Size()); };
		Weights.push_back(Eigen::MatrixXf::NullaryExpr(Layers[i]->Size(),Layers[i - 1]->Size(), normalWeight));
	}

	std::cout << "Initialization finished" << std::endl;
}

std::tuple <int, float> Network::Evaluate()
{
	int goodResult = 0;
	float sumError = 0;

	for (int i = 0; i < std::get<0>(TestData).size(); i++)
	{
		// Todo call function to convert opencv mat to eigen vec
		cv::Mat imageCV = std::get<0>(TestData)[i];
		Eigen::VectorXf image = cv2eigen::Convert(imageCV);

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

		float error = Function::ErrorFunction(_Loss, ConvertLabel2LastLayer(std::get<1>(TestData)[i]), L).sum();
		sumError += error;

		if (std::get<1>(TestData)[i] == maxIndex)
		{
			goodResult++;
		}

		std::cout << "\r                                               ";
		std::cout << "\rEvaluation.  Acc: " + std::to_string(goodResult) << " / " << std::to_string(i) << " Loss: " << std::setprecision(2) << error;
	}
	sumError = sumError / std::get<0>(TestData).size();
	
	return { goodResult, sumError };
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
		
		//// change place
		//if (i % 4 == 0) 
		//{
		//	DataAugmentation::Rotate(imageCV, 30);
		//}

		int label = std::get<1>(batch)[i];
		auto errorWeightsBiases = BackPropagation(cv2eigen::Convert(imageCV), label);

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

	// this needs to be more generic lol
	/*Eigen::VectorXf costPrime = Function::ErrorFunctionPrime(_Loss, ConvertLabel2LastLayer(label), activations.back());
	Eigen::VectorXf sigmoidPrime = Function::ActivationFunctionPrime(Layers.back()->_Activation, zs.back());
	Eigen::VectorXf delta = costPrime.array() * sigmoidPrime.array();*/
	Eigen::VectorXf delta = Function::DeltaLastLayer(_Loss, ConvertLabel2LastLayer(label), activations.back(), zs.back());

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
