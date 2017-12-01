#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include "HoNC\HoNC.h"


#include <vector>
using namespace std;

using namespace cv;
using namespace cv::xfeatures2d;
/*
FeatureExtractor
This class abstracts out the process of extracting the features
of an image. The user simply picks an extraction algorithm, and the
class works out the details of implementation on its own.
*/

typedef enum
{
	E_None,
	E_SIFT,
	E_SURF,
	E_HoNC
} ExtractType;

ExtractType ExtractTypeFromString(const string &str);

class FeatureExtractor
{
public:
	FeatureExtractor();
	~FeatureExtractor();

	void ExtractFeatures(ExtractType type, const Mat& img, vector<KeyPoint> &keypoints, Mat &descriptors);
	
	bool GetExtractor(ExtractType type, Ptr<DescriptorExtractor> &extractor);

private:

	Ptr<SIFT> sift;
	Ptr<SURF> surf;
	Ptr<HoNC> honc;

	void RunSIFT(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
	void RunSURF(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
	void RunHoNC(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
};

