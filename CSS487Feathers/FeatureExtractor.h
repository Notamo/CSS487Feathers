#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include "HoNC\HoNC.h"

#include "FeatherIDUtil.h"

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

//Extraction methods
typedef enum
{
	E_None,
	E_SIFT,
	E_SURF,
	E_HoNC
} ExtractType;

bool ExtractTypeFromString(const string &str, ExtractType &type);

class FeatureExtractor
{
public:
	FeatureExtractor();
	~FeatureExtractor();

	void Detect(ExtractType type, const Mat& img, vector<KeyPoint> &keypoints);
	void Compute(ExtractType type, const Mat& img, vector<KeyPoint> &keypoints, Mat &descriptors);
	void ExtractFeatures(ExtractType type, const Mat& img, vector<KeyPoint> &keypoints, Mat &descriptors);
	
	bool FeatureExtractor::GetFD(ExtractType type, Ptr<FeatureDetector> &FD);
	bool FeatureExtractor::GetDE(ExtractType type, Ptr<DescriptorExtractor> &DE);

private:

	Ptr<SIFT> sift;
	Ptr<SURF> surf;
	Ptr<HoNC> honc;

	void DetectSIFT(const Mat &img, vector<KeyPoint> &keypoints);
	void DetectSURF(const Mat &img, vector<KeyPoint> &keypoints);
	void DetectHoNC(const Mat &img, vector<KeyPoint> &keypoints);

	void ComputeSIFT(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
	void ComputeSURF(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
	void ComputeHoNC(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
};

