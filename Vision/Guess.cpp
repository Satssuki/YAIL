#include "Guess.h"



Guess::Guess()
{
	
}


Guess::~Guess()
{
}

void Guess::StartTest(std::vector<cv::Mat>* images, std::vector<int>* labels)
{
	std::cout << "Try to guess if the image contains a cat (press 0) or a dog (press 1)." << std::endl;
	std::cout << "Enter the number of images you want to test" << std::endl;
	
	int totalImages;
	std::cin >> totalImages;
	if (totalImages < 0 || totalImages > images->size()) 
	{
		std::cout << "Error number images" << std::endl;
		return;
	}

	// the test is easy because the images are resized
	int goodAnswer = 0;
	for (int i = 0; i < totalImages; i++)
	{
		cv::Mat out = (*images)[i];
		resize((*images)[i], out, cv::Size(200, 200));
		cv::imshow("Guess Test", out);

		char key = cv::waitKey(0);
		char answer = (char)((*labels)[i] + 48);
		if (key == answer)
		{
			goodAnswer++;
			std::cout << "Good answer!" << std::endl;
		} else { std::cout << "Wrong answer!" << std::endl; }
	}

	std::cout << "You had " << goodAnswer << "/" << totalImages << " good answers!" << std::endl;
}

void Guess::Print(int label, int result, cv::Mat image)
{
	cv::Mat resized;
	resize(image, resized, cv::Size(200, 200));
	cvtColor(resized, resized, CV_GRAY2BGR);

	// tried to make them static
	cv::Scalar Red = cv::Scalar(0, 0, 1);
	cv::Scalar Green = cv::Scalar(0, 1, 0);

	int top = (int)(0.05*resized.rows), bottom = (int)(0.05*resized.rows);
	int left = (int)(0.05*resized.cols), right = (int)(0.05*resized.cols);

	if(label == result)
		copyMakeBorder(resized, resized, top, bottom, left, right, cv::BORDER_CONSTANT, Green);
	else
		copyMakeBorder(resized, resized, top, bottom, left, right, cv::BORDER_CONSTANT, Red);

	std::cout << " Prediction: " << result << " Label: " << label << std::endl;

	imshow("Prediction", resized);
	//cv::waitKey(0);
}
