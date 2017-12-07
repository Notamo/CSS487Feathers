#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

#include "FeatherIDUtil.h"

#include <vector>
using namespace std;

using namespace cv;
using namespace cv::xfeatures2d;
/*
FeatureExtractor
This class allows the user to retrieve any
descriptor or extractor we add
*/

//Extraction methods
typedef enum
{
	E_None,
	E_SIFT,
	E_SURF
} ExtractType;

bool ExtractTypeFromString(const string &str, ExtractType &type);
string StringFromExtractType(ExtractType type);

class FeatureExtractor
{
public:
	FeatureExtractor();
	~FeatureExtractor();
	
	bool FeatureExtractor::GetFD(ExtractType type, Ptr<FeatureDetector> &FD);
	bool FeatureExtractor::GetDE(ExtractType type, Ptr<DescriptorExtractor> &DE);

private:
	Ptr<SIFT> sift;
	Ptr<SURF> surf;
};

