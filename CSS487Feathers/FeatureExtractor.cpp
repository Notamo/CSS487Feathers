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
void FeatureExtractor::ExtractFeatures(ExtractType type, const Mat& img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	switch (type)
	{
		case ExtractType::E_SIFT:
		{
			RunSIFT(img, keypoints, descriptors);
			break;
		}
		case ExtractType::E_SURF:
		{
			RunSURF(img, keypoints, descriptors);
			break;
		}
		case ExtractType::E_HoNC:
		{
			RunHoNC(img, keypoints, descriptors);
			break;
		}
	}
}

void FeatureExtractor::RunSIFT(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	Mat greyImg;
	cvtColor(img, greyImg, COLOR_BGR2GRAY);

	sift->detect(img, keypoints);
	sift->compute(greyImg, keypoints, descriptors);
}

void FeatureExtractor::RunSURF(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	Mat greyImg;
	cvtColor(img, greyImg, COLOR_BGR2GRAY);

	surf->detect(img, keypoints);
	surf->compute(greyImg, keypoints, descriptors);
}

void FeatureExtractor::RunHoNC(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	(*honc)(img, noArray(), keypoints, descriptors, false);
}

ExtractType ExtractTypeFromString(const string &str)
{
	if (str == "SIFT")
	{
		return ExtractType::E_SIFT;
	}
	else if (str == "SURF")
	{
		return ExtractType::E_SURF;
	}
	else if (str == "HoNC")
	{
		return ExtractType::E_HoNC;
	}
	else
	{
		return ExtractType::E_None;
	}
}