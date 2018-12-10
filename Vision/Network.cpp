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

	// Todo convert image and label to eigen matrix and vector
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

	int trainDataSize = std::get<0>(TrainData).size();
	int totalBatches = ceil(trainDataSize / (float)BatchSize);

	for (int e = 0; e < Epoch; e++)
	{
		// shuffle data
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		shuffle(std::get<0>(TrainData).begin(), std::get<0>(TrainData).end(), std::default_random_engine(seed));
		shuffle(std::get<1>(TrainData).begin(), std::get<1>(TrainData).end(), std::default_random_engine(seed));
	
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

			// Todo pass starting index and size, because slicing is O(n) 
			UpdateBatch({ batchImages, batchLabels });

			if (b % 100 == 0) { std::cout << "\rBatch " << b << " / " << totalBatches; }
		}
		
		std::string path = "Backup/Epoch_" + std::to_string(e) + ".dat";
		SaveWeights(path);

		auto evaluation = Evaluate();
		std::cout << "\r                     ";
		std::cout << "\rEpoch " << e << ". Acc : " + std::to_string(std::get<0>(evaluation)) << " / " << std::get<0>(TestData).size() << " Loss: " << std::get<1>(evaluation) << std::endl;
	}

	clock_t end = clock() - start;
	double secs = end / (double)CLOCKS_PER_SEC;
	float min = fmod(secs / 60, 60);

	std::cout << "Training completed. Took: " << min << " min" << std::endl;
}

int Network::Predict(cv::Mat image)
{
	Eigen::VectorXf imageEigen = VectorizeImage(image);
	Eigen::VectorXf L = Forward(imageEigen);
	float max = L(0);
	int maxIndex = 0;
	for (int v = 1; v < L.rows(); v++)
	{
		if (max < L(v))
		{
			max = L(v);
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

void Network::NormalInitialization()
{
	std::cout << "Initialize weights and biases" << std::endl;

	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, 1.0);
	
	auto normalBias = [&](float) {return distribution(generator); };

	Biases.reserve(Layers.size() - 1);
	Weights.reserve(Layers.size() - 1);

	for(int i = 1; i < Layers.size(); i++) 
	{
		Biases.push_back(Eigen::VectorXf::NullaryExpr(Layers[i]->Size(), normalBias));	

		auto normalWeight = [&](float) {return distribution(generator)/sqrt(Layers[i - 1]->Size()); };
		Weights.push_back(Eigen::MatrixXf::NullaryExpr(Layers[i]->Size(), Layers[i - 1]->Size(), normalWeight));
	}

	std::cout << "Initialization finished" << std::endl;
}

std::tuple <int, float> Network::Evaluate()
{
	int goodResult = 0;
	float sumError = 0;

	for (int i = 0; i < std::get<0>(TestData).size(); i++)
	{
		// Todo use already converted image
		cv::Mat imageCV = std::get<0>(TestData)[i];
		Eigen::VectorXf image = VectorizeImage(imageCV);

		Eigen::VectorXf L = Forward(image);
		float max = L(0);
		int maxIndex = 0;
		for (int v = 1; v < L.rows(); v++)
		{
			if (max < L(v))
			{
				max = L(v);
				maxIndex = v;
			}
		}

		// Todo use already vectorized label
		float error = Function::ErrorFunction(_Loss, VectorizeLabel(std::get<1>(TestData)[i]), L).sum();
		sumError += error;

		if (std::get<1>(TestData)[i] == maxIndex)
		{
			goodResult++;
		}

		std::cout << "\r                                               ";
		std::cout << "\rEvaluation.  Acc: " + std::to_string(goodResult) << " / " << std::to_string(i) << " Loss: " << error;
	}

	sumError = sumError / std::get<0>(TestData).size();
	
	return { goodResult, sumError };
}

void Network::UpdateBatch(std::tuple<std::vector<cv::Mat>, std::vector<int>> batch)
{
	int currentBatchSize = std::get<0>(batch).size();

	std::vector<Eigen::MatrixXf> sumWeightsError;
	sumWeightsError.reserve(Layers.size() - 1);

	std::vector<Eigen::VectorXf> sumBiasesError;
	sumBiasesError.reserve(Layers.size() - 1);
	
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		sumWeightsError.push_back(Eigen::MatrixXf::Zero(Weights[i].rows(), Weights[i].cols())); 
		sumBiasesError.push_back(Eigen::VectorXf::Zero(Biases[i].rows()));
	}

	for (int i = 0; i < currentBatchSize; i++)
	{
		// Todo use already converted eigen image and label
		cv::Mat imageCV = std::get<0>(batch)[i];
		
		int label = std::get<1>(batch)[i];
		auto errorWeightsBiases = BackPropagation(VectorizeImage(imageCV), label);

		for (int iL = 0; iL < Layers.size() - 1; iL++)
		{
			sumWeightsError[iL].array() += std::get<0>(errorWeightsBiases)[iL].array();
			sumBiasesError[iL].array() += std::get<1>(errorWeightsBiases)[iL].array();
		}	
	}
	
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		Weights[i].array() -= (LearningRate / currentBatchSize) * sumWeightsError[i].array();
		Biases[i].array() -= (LearningRate / currentBatchSize) * sumBiasesError[i].array();
	}
}

std::tuple<std::vector<Eigen::MatrixXf>, std::vector<Eigen::VectorXf>> Network::BackPropagation(Eigen::VectorXf image, int label)
{
	std::vector< Eigen::VectorXf> activations;
	activations.reserve(Layers.size());
	activations.push_back(image);

	std::vector< Eigen::VectorXf> zs;
	zs.reserve(Layers.size() - 1);

	Eigen::VectorXf a = image;
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		Eigen::VectorXf z = Weights[i] * a + Biases[i];
		a = Function::ActivationFunction(Layers[i + 1]->_Activation, z);
		
		zs.push_back(z);
		activations.push_back(a);
	}

	std::vector< Eigen::MatrixXf> deltaWeights;
	deltaWeights.reserve(Layers.size() - 1);

	std::vector< Eigen::VectorXf> deltaBiases;
	deltaBiases.reserve(Layers.size() - 1);

	// Todo use already converted label
	Eigen::VectorXf delta = Function::DeltaLastLayer(_Loss, VectorizeLabel(label), activations.back(), zs.back());

	deltaBiases.push_back(delta);
	deltaWeights.push_back(delta * activations[activations.size() - 2].transpose());

	for (int i = 2; i < Layers.size(); i++)
	{
		Eigen::VectorXf fp = Function::ActivationFunctionPrime(Layers[Layers.size() - i]->_Activation, zs[zs.size() - i]);
		delta = (Weights[Weights.size() - i + 1].transpose() * delta).array() * fp.array();

		// Todo replace insert by []
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

Eigen::VectorXf Network::VectorizeLabel(int label)
{
	Eigen::VectorXf lastLayer = Eigen::VectorXf::Zero(Layers.back()->Size());
	lastLayer(label) = 1;

	return lastLayer;
}

Eigen::VectorXf Network::VectorizeImage(cv::Mat imageCV)
{
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(reinterpret_cast<float*>(imageCV.data), imageCV.rows, imageCV.cols * imageCV.channels());
	Eigen::Map<Eigen::RowVectorXf> image(mat.data(), mat.size());

	return image;
}
