#pragma once

#include <vector>
#include <string>
using namespace std;

#include <opencv2\core.hpp>
using namespace cv;

const int MIN_FEATHER_WORDS = 10;

typedef struct
{
	string name;
	float rating = 0.0f;
} RatingPair;

typedef struct
{
	vector<Mat> images;
	string name;
	int label;
} ImageSet;

