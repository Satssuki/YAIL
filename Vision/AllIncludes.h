#pragma once

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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

// our classes

#include "GuessTest.h"
#include "Network.h"

// data
#include "DataLoader.h"
#include "DataAugmentation.h"

// layer
#include "Layer.h"
#include "Dense.h"


// enum
#include "Loss.h"
#include "Optimizer.h"
#include "Activation.h"