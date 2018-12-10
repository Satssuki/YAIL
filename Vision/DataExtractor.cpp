#include "DataExtractor.h"

std::vector<cv::Mat> DataExtractor::ExtractCharacters(cv::Mat & input) {
	cv::Mat temp, gray;
	std::vector<cv::Mat> resultArray;
	cvtColor(input, gray, CV_BGR2GRAY);
	gray = gray > 127;

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(gray, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	std::sort(contours.begin(), contours.end(), contour_sorter());
	cv::Rect minRect;

	if (contours.size() > 0) {
		for (size_t i = 0; i < contours.size(); i++) {
			minRect = cv::boundingRect(cv::Mat(contours[i]));
			if (minRect.width > input.cols * 0.9) {
				minRect.height = minRect.y + 4;
				minRect.y = 0;
				temp = input(minRect);
				temp = cv::Scalar(255, 255, 255);
				resultArray = StepCharacter(input);
			}
		}
	}
	return resultArray;
}

std::vector<cv::Mat> DataExtractor::StepCharacter(cv::Mat & input) {
	cv::Mat gray;
	std::vector<cv::Mat> characArray;
	cvtColor(input, gray, CV_BGR2GRAY);
	threshold(gray, gray, 200, 255, cv::THRESH_BINARY_INV);
	RNG rng(12345);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	findContours(gray, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	std::sort(contours.begin(), contours.end(), contour_sorter());
	int padding = 5;
	for (int i = 0; i < contours.size(); i = hierarchy[i][0])
	{
		cv::Mat charac;
		cv::Rect r = boundingRect(contours[i]);
		rectangle(input, cv::Point(r.x - padding, r.y), cv::Point(r.x + r.width, r.y + r.height), Scalar(0, 0, 255), 1, 8, 0);
		gray(cv::Rect(cv::Point(r.x - padding, r.y), cv::Point(r.x + r.width, r.y + r.height))).copyTo(charac);
		cv::copyMakeBorder(charac, charac, 20, 20, 20, 20, BORDER_CONSTANT, cv::Scalar(0,0,0));
		cv::resize(charac, charac, cv::Size(28, 28));
		cv::waitKey(0);
		characArray.push_back(charac);
	}
	return characArray;
}
