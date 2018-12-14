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
	
	/********* Test *********/
	for (int i = 0; i < 10; i++) {
		auto normalWeight = [&](float) {return distribution(generator) / sqrt(5 * 5); };
		Conv1.push_back(Eigen::MatrixXf::NullaryExpr(5, 5, normalWeight));
	}
	for (int i = 0; i < 10; i++) {
		auto normalWeight = [&](float) {return distribution(generator) / sqrt(5 * 5); };
		Conv2.push_back(Eigen::MatrixXf::NullaryExpr(5, 5, normalWeight));
	}
	/************************/

	auto normalBias = [&](float) {return distribution(generator); };

	//Biases.reserve(Layers.size() - 1);
	//Weights.reserve(Layers.size() - 1);

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
		
		//Eigen::VectorXf image = VectorizeImage(imageCV);
		//Eigen::VectorXf L = Forward(image);
		Eigen::MatrixXf mat = MatrixImage(imageCV);
		Eigen::VectorXf L = ForwardCNN(mat);

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
		//auto errorWeightsBiases = BackPropagation(VectorizeImage(imageCV), label);
		auto errorWeightsBiases = BackPropagation(MatrixImage(imageCV), label);

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

		for (int n = 0; n < 10; n++)
		{
			Conv1[i].array() -= (LearningRate / currentBatchSize) * Conv1dKerneld[i].array();
			Conv2[i].array() -= (LearningRate / currentBatchSize) * Conv2dKerneld[i].array();
		}
	}
	Conv1dKerneld.shrink_to_fit();
	Conv2dKerneld.shrink_to_fit();
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

float Relu(float x)
{
	return max(0.0f, x);
}

std::tuple<std::vector<Eigen::MatrixXf>, std::vector<Eigen::VectorXf>> Network::BackPropagation(Eigen::MatrixXf image, int label)
{
	//std::cout << "Image" << image.format(CleanFmt) << std::endl;

	// conv first layer
	vector<Eigen::MatrixXf> conv1HS;
	for (int i = 0; i < 10; i++)
	{
		//std::cout << "Conv" << Conv1[i].format(CleanFmt) << std::endl;

		Eigen::MatrixXf h(24, 24);
		for (int y = 0; y < 24; y++) 
		{
			for (int x = 0; x < 24; x++)
			{
					// convolution with a stride of 1
				Eigen::MatrixXf block = image.block<5, 5>(y, x).array();
				//std::cout << block.format(CleanFmt) << std::endl;

				h(y,x) = (block.array() * Conv1[i].array()).sum();
			}
		}

		// Todo apply relu activation function
		h = h.unaryExpr(&Relu);

		//std::cout << h.format(CleanFmt) << std::endl;

		conv1HS.push_back(h);
	}

	//std::cout << "Conv layer 1" << conv1HS.back().format(CleanFmt) << std::endl;

	// mean pool with a stride of 2
	vector<Eigen::MatrixXf> maxPoolHS;
	for (int i = 0; i < 10; i++)
	{
		Eigen::MatrixXf poolH(12, 12);
		for (int y = 0; y < 12; y++)
		{
			for (int x = 0; x < 12; x++)
			{
				Eigen::MatrixXf block = conv1HS[i].block<2, 2>(y * 2, x * 2).array();
				//std::cout << block.format(CleanFmt) << std::endl;

				//poolH(y, x) = block.maxCoeff();
				poolH(y, x) = block.mean();
			}
		}

		//std::cout << poolH.format(CleanFmt) << std::endl;
		maxPoolHS.push_back(poolH);
	}

	//std::cout << "MaxPool layer 1" << maxPoolHS.back().format(CleanFmt) << std::endl;

	// conv second layer
	vector<Eigen::MatrixXf> conv2HS;
	for (int i = 0; i < 10; i++)
	{
		//std::cout << "Conv" << Conv2[i].format(CleanFmt) << std::endl;

		Eigen::MatrixXf h(8, 8);
		for (int y = 0; y < 8; y++)
		{
			for (int x = 0; x < 8; x++)
			{
				// convolution with a stride of 1
				Eigen::MatrixXf block = maxPoolHS[i].block<5, 5>(y, x).array();
				//std::cout << block.format(CleanFmt) << std::endl;

				h(y, x) = (block.array() * Conv2[i].array()).sum();
			}
		}

		// Todo apply activation function
		h = h.unaryExpr(&Relu);

		//std::cout << h.format(CleanFmt) << std::endl;
		conv2HS.push_back(h);
	}

	//std::cout << "Conv layer 2" << conv2HS.back().format(CleanFmt) << std::endl;

	// mean pool with a stride of 2
	vector<Eigen::MatrixXf> maxPoolHS2;
	for (int i = 0; i < 10; i++)
	{
		Eigen::MatrixXf poolH(4, 4);
		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				Eigen::MatrixXf block = conv2HS[i].block<2, 2>(y * 2, x * 2).array();
				//std::cout << block.format(CleanFmt) << std::endl;

				//poolH(y, x) = block.maxCoeff();
				poolH(y, x) = block.mean();
			}
		}

		//std::cout << poolH.format(CleanFmt) << std::endl;
		maxPoolHS2.push_back(poolH);
	}

	//std::cout << "MaxPool layer 2" << maxPoolHS2.back().format(CleanFmt) << std::endl;

	// flatten layer
	Eigen::MatrixXf mat2vec(10, 4 * 4);
	Eigen::VectorXf flatten(10 * 4 * 4);

	for (int i = 0; i < 10; i++)
	{
		Eigen::Map<Eigen::RowVectorXf> vec(maxPoolHS2[i].data(), maxPoolHS2[i].size());
		mat2vec.row(i) = vec;
	}

	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> preFlatten(mat2vec);
	flatten = Eigen::Map<Eigen::RowVectorXf>(preFlatten.data(), preFlatten.size());

	// normal forward pass
	std::vector< Eigen::VectorXf> activations;
	activations.push_back(flatten);

	std::vector< Eigen::VectorXf> zs;
	zs.reserve(Layers.size() - 1);

	Eigen::VectorXf a = flatten;
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		Eigen::VectorXf z = Weights[i] * a + Biases[i];
		a = Function::ActivationFunction(Layers[i + 1]->_Activation, z);

		zs.push_back(z);
		activations.push_back(a);
	}

	// backpropagation
	std::vector< Eigen::MatrixXf> deltaWeights;
	std::vector< Eigen::VectorXf> deltaBiases;

	Eigen::VectorXf delta = Function::DeltaLastLayer(_Loss, VectorizeLabel(label), activations.back(), zs.back());

	deltaBiases.push_back(delta);
	deltaWeights.push_back(delta * activations[activations.size() - 2].transpose());

	for (int i = 2; i < Layers.size(); i++)
	{
		Eigen::VectorXf fp = Function::ActivationFunctionPrime(Layers[Layers.size() - i]->_Activation, zs[zs.size() - i]);
		delta = (Weights[Weights.size() - i + 1].transpose() * delta).array() * fp.array();

		deltaBiases.insert(deltaBiases.begin(), delta);
		deltaWeights.insert(deltaWeights.begin(), delta * activations[activations.size() - i - 1].transpose());
	}

	/*** todo ***/
	// backprop on maxpool and conv layer
	
	// partial derivatives of input neurons, is the same as the partial derivatives of the bias?
	Eigen::VectorXf fp = Function::ActivationFunctionPrime(sigmoid, flatten); // values of the flatten are already activated
	delta = (Weights[0].transpose() * delta).array() * fp.array();

	std::vector<Eigen::MatrixXf> conv2DHZ;
	// convert vec to multiple array
	for (int i = 0; i < 10; i++)
	{
		Eigen::VectorXf subVec = delta.array().segment(i * 16, 16); 
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> subMat(reinterpret_cast<float*>(subVec.data()), 4, 4);

		// divide error for mean pooling 
		Eigen::MatrixXf mat(8, 8);

		subMat = subMat.array() / 4;

		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				mat.block<2, 2>(y * 2, x * 2) = Eigen::Matrix2f::Constant(subMat(y, x));
			}
		}
		// expend
		conv2DHZ.push_back(mat);
	}

	std::vector<Eigen::MatrixXf> max1dhz;
	for (int i = 0; i < 10; i++)
	{
		Eigen::MatrixXf mat(12, 12);
		Eigen::MatrixXf kernel(5, 5);
		for (int y = 0; y < 8; y++)
		{
			for (int x = 0; x < 8; x++)
			{
				mat.block<5, 5>(y, x).array() += Conv2[i].array() * conv2DHZ[i](y, x);
				kernel.array() += mat.block<5, 5>(y, x).array() * conv2DHZ[i](y, x);
			}
		}
		max1dhz.push_back(mat);
		Conv2dKerneld.push_back(kernel);
	}	

	std::vector<Eigen::MatrixXf> conv1DHZ;
	for (int i = 0; i < 10; i++)
	{
		Eigen::MatrixXf subMat(12, 12);
		subMat.array() = max1dhz[i].array() / 4;

		Eigen::MatrixXf mat(24, 24);

		for (int y = 0; y < 12; y++)
		{
			for (int x = 0; x < 12; x++)
			{
				mat.block<2, 2>(y * 2, x * 2) = Eigen::Matrix2f::Constant(subMat(y, x));
			}
		}
		conv1DHZ.push_back(mat);
	}

	for (int i = 0; i < 10; i++)
	{
		Eigen::MatrixXf mat(28, 28);
		Eigen::MatrixXf kernel(5, 5);
		for (int y = 0; y < 24; y++)
		{
			for (int x = 0; x < 24; x++)
			{
				mat.block<5, 5>(y, x).array() += Conv1[i].array() * conv1DHZ[i](y, x);
				kernel.array() += mat.block<5, 5>(y, x).array() * conv1DHZ[i](y, x);
			}			
		}
		Conv1dKerneld.push_back(kernel);
	}	

	// convert vector back to multiple matrix
	return { deltaWeights, deltaBiases };
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

Eigen::VectorXf Network::ForwardCNN(Eigen::MatrixXf input)
{
	vector<Eigen::MatrixXf> conv1HS;
	for (int i = 0; i < 10; i++)
	{
		//std::cout << "Conv" << Conv1[i].format(CleanFmt) << std::endl;

		Eigen::MatrixXf h(24, 24);
		for (int y = 0; y < 24; y++)
		{
			for (int x = 0; x < 24; x++)
			{
				// convolution with a stride of 1
				Eigen::MatrixXf block = input.block<5, 5>(y, x).array();
				//std::cout << block.format(CleanFmt) << std::endl;

				h(y, x) = (block.array() * Conv1[i].array()).sum();
			}
		}

		//std::cout << h.format(CleanFmt) << std::endl;
		conv1HS.push_back(h);
	}

	//std::cout << "Conv layer 1" << conv1HS.back().format(CleanFmt) << std::endl;

	// max pool with a stride of 2
	vector<Eigen::MatrixXf> maxPoolHS;
	for (int i = 0; i < 10; i++)
	{
		Eigen::MatrixXf poolH(12, 12);
		for (int y = 0; y < 12; y++)
		{
			for (int x = 0; x < 12; x++)
			{
				Eigen::MatrixXf block = conv1HS[i].block<2, 2>(y * 2, x * 2).array();
				//std::cout << block.format(CleanFmt) << std::endl;

				poolH(y, x) = block.maxCoeff();
			}
		}

		//std::cout << poolH.format(CleanFmt) << std::endl;
		maxPoolHS.push_back(poolH);
	}

	//std::cout << "MaxPool layer 1" << maxPoolHS.back().format(CleanFmt) << std::endl;

	// conv second layer
	vector<Eigen::MatrixXf> conv2HS;
	for (int i = 0; i < 10; i++)
	{
		//std::cout << "Conv" << Conv2[i].format(CleanFmt) << std::endl;

		Eigen::MatrixXf h(8, 8);
		for (int y = 0; y < 8; y++)
		{
			for (int x = 0; x < 8; x++)
			{
				// convolution with a stride of 1
				Eigen::MatrixXf block = maxPoolHS[i].block<5, 5>(y, x).array();
				//std::cout << block.format(CleanFmt) << std::endl;

				h(y, x) = (block.array() * Conv2[i].array()).sum();
			}
		}

		//std::cout << h.format(CleanFmt) << std::endl;
		conv2HS.push_back(h);
	}

	//std::cout << "Conv layer 2" << conv2HS.back().format(CleanFmt) << std::endl;

	// max pool with a stride of 2
	vector<Eigen::MatrixXf> maxPoolHS2;
	for (int i = 0; i < 10; i++)
	{
		Eigen::MatrixXf poolH(4, 4);
		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				Eigen::MatrixXf block = conv2HS[i].block<2, 2>(y * 2, x * 2).array();
				//std::cout << block.format(CleanFmt) << std::endl;

				poolH(y, x) = block.maxCoeff();
			}
		}

		//std::cout << poolH.format(CleanFmt) << std::endl;
		maxPoolHS2.push_back(poolH);
	}

	//std::cout << "MaxPool layer 2" << maxPoolHS2.back().format(CleanFmt) << std::endl;

	// flatten layer
	Eigen::MatrixXf mat2vec(10, 4 * 4);
	Eigen::VectorXf flatten(10 * 4 * 4);

	for (int i = 0; i < 10; i++)
	{
		Eigen::Map<Eigen::RowVectorXf> vec(maxPoolHS2[i].data(), maxPoolHS2[i].size());
		mat2vec.row(i) = vec;
	}

	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> preFlatten(mat2vec);
	flatten = Eigen::Map<Eigen::RowVectorXf>(preFlatten.data(), preFlatten.size());

	// normal forward pass

	Eigen::VectorXf a = flatten;
	for (int i = 0; i < Layers.size() - 1; i++)
	{
		Eigen::VectorXf z = Weights[i] * a + Biases[i];
		a = Function::ActivationFunction(Layers[i + 1]->_Activation, z);
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

Eigen::MatrixXf Network::MatrixImage(cv::Mat imageCV)
{
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(reinterpret_cast<float*>(imageCV.data), imageCV.rows, imageCV.cols * imageCV.channels());

	return mat;
}