#include "FeatureExtractor.h"


FeatureExtractor::FeatureExtractor()
{
	sift = SIFT::create();
	surf = SURF::create();
	honc = HoNC::create();
}


FeatureExtractor::~FeatureExtractor()
{
}

//extract features based on the algorithm chosen
//later we might even combine them
void FeatureExtractor::ExtractFeatures(ExtractType type, const Mat& img, vector<KeyPoint> &keypoints, vector<Mat> &descriptors)
{
	switch (type)
	{
		case ExtractType::E_SIFT:
		{
			sift->detectAndCompute(img, noArray(), keypoints, descriptors, false);
			break;
		}
		case ExtractType::E_SURF:
		{
			surf->detectAndCompute(img, noArray(), keypoints, descriptors, false);
			break;
		}
		case ExtractType::E_HoNC:
		{
			(*honc)(img, noArray(), keypoints, descriptors, false);
			break;
		}
	}
}