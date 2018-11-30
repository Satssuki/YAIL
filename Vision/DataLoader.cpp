#include "DataLoader.h"



DataLoader::DataLoader()
{
	// we use the same seed to order the images and labels together
	Seed = std::chrono::system_clock::now().time_since_epoch().count();
}


DataLoader::~DataLoader()
{
}

std::tuple<std::vector<cv::Mat>, std::vector<int>> DataLoader::LoadData(std::string rootPath)
{
	std::vector<cv::Mat> testImages;
	std::vector<int> testLabels;

	LoadImages(rootPath + "/cat/*.jpg", &testImages);
	testLabels.insert(testLabels.end(), testImages.size(), 0);

	int catSize = testImages.size();

	LoadImages(rootPath + "/dog/*.jpg", &testImages);
	testLabels.insert(testLabels.end(), testImages.size() - catSize, 1);

	shuffle(testImages.begin(), testImages.end(), std::default_random_engine(Seed));
	shuffle(testLabels.begin(), testLabels.end(), std::default_random_engine(Seed));

	return { testImages, testLabels };
}

void DataLoader::LoadImages(std::string path, std::vector<cv::Mat>* images)
{
	std::vector<cv::String> fn;
	glob(path, fn, false);
	size_t count = fn.size();
	for (size_t i = 0; i < count; i++)
	{
		cv::Mat image = imread(fn[i]);

		if (&image != nullptr && image.data) {
			cv::resize(image, image, Size);
			image.convertTo(image, CV_32FC3, 1.f / 255); // normalize pixel values between 0 and 1

			images->push_back(image);			
		}
	}
}

