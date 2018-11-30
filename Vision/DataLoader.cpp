#include "DataLoader.h"



DataLoader::DataLoader()
{
	// we use the same seed to order the images and labels together
	Seed = std::chrono::system_clock::now().time_since_epoch().count();
}


DataLoader::~DataLoader()
{
}

std::tuple<std::vector<cv::Mat*>, std::vector<uint8_t>> DataLoader::LoadTestData()
{
	return std::tuple<std::vector<cv::Mat*>, std::vector<uint8_t>>();
}

std::tuple<std::vector<cv::Mat*>, std::vector<uint8_t>> DataLoader::LoadTrainData()
{
	return std::tuple<std::vector<cv::Mat*>, std::vector<uint8_t>>();
}
