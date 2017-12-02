#pragma once

#include <vector>
#include <string>
using namespace std;

#include <opencv2\core.hpp>
using namespace cv;

typedef struct
{
	string name;
	float rating = 0.0f;
} RatingPair;

typedef struct
{
	vector<Mat> images;
	string name;
} TrainingSet;