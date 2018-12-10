#pragma once

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

// eigen
#include <iostream>
#include <Eigen/Dense>

// others
#include <iostream>
#include <tuple>
#include <cmath>
#include <vector>
#include <algorithm>    
#include <array>       
#include <random>       
#include <chrono>    
#include <string>
#include <time.h>
#include <iomanip>
#include <math.h>   

// our classes
#include "Guess.h"

// data
#include "DataLoader.h"
#include "DataAugmentation.h"
#include "MNIST.h"
#include "EigenSerializer.h"
#include "DataExtractor.h"
#include "Network.h"

//Static function
#include "Function.h"