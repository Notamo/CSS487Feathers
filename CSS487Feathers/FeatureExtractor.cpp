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

bool FeatureExtractor::GetFD(ExtractType type, Ptr<FeatureDetector> &FD)
{
	switch (type)
	{
		case ExtractType::E_SIFT:
		{
			FD = sift;
			return true;
		}
		case ExtractType::E_SURF:
		{
			FD = surf;
			return true;
		}
		case ExtractType::E_HoNC:
		{
			FD = honc;
			return true;
		}
	}

	return false;
}

bool FeatureExtractor::GetDE(ExtractType type, Ptr<DescriptorExtractor> &DE)
{
	switch (type)
	{
		case ExtractType::E_SIFT:
		{
			DE = sift;
			return true;
		}
		case ExtractType::E_SURF:
		{
			DE = surf;
			return true;
		}
		case ExtractType::E_HoNC:
		{
			DE = honc;
			return true;
		}
	}

	return false;
}

bool ExtractTypeFromString(const string &str, ExtractType &type)
{
	if (str == "SIFT")
	{
		type =  ExtractType::E_SIFT;
	}
	else if (str == "SURF")
	{
		type = ExtractType::E_SURF;
	}
	else if (str == "HoNC")
	{
		type = ExtractType::E_HoNC;
	}
	else
	{
		return false;
	}

	return true;
}