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

void FeatureExtractor::Detect(ExtractType type, const Mat& img, vector<KeyPoint> &keypoints)
{
	switch (type)
	{
		case ExtractType::E_SIFT:
		{
			DetectSIFT(img, keypoints);
			break;
		}
		case ExtractType::E_SURF:
		{
			DetectSURF(img, keypoints);
			break;
		}
		case ExtractType::E_HoNC:
		{
			DetectHoNC(img, keypoints);
			break;
		}
	}
}

void FeatureExtractor::Compute(ExtractType type, const Mat& img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	switch (type)
	{
		case ExtractType::E_SIFT:
		{
			ComputeSIFT(img, keypoints, descriptors);
			break;
		}
		case ExtractType::E_SURF:
		{
			ComputeSURF(img, keypoints, descriptors);
			break;
		}
		case ExtractType::E_HoNC:
		{
			ComputeHoNC(img, keypoints, descriptors);
			break;
		}
	}
}

//extract features based on the algorithm chosen
//later we might even combine them
void FeatureExtractor::ExtractFeatures(ExtractType type, const Mat& img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	switch (type)
	{
		case ExtractType::E_SIFT:
		{
			DetectSIFT(img, keypoints);
			ComputeSIFT(img, keypoints, descriptors);
			break;
		}
		case ExtractType::E_SURF:
		{
			DetectSURF(img, keypoints);
			ComputeSURF(img, keypoints, descriptors);
			break;
		}
		case ExtractType::E_HoNC:
		{
			DetectHoNC(img, keypoints);
			ComputeHoNC(img, keypoints, descriptors);
			break;
		}
	}
}

void FeatureExtractor::DetectSIFT(const Mat &img, vector<KeyPoint> &keypoints)
{
	Mat greyImg;
	cvtColor(img, greyImg, COLOR_BGR2GRAY);

	sift->detect(img, keypoints);
}

void FeatureExtractor::DetectSURF(const Mat &img, vector<KeyPoint> &keypoints)
{
	Mat greyImg;
	cvtColor(img, greyImg, COLOR_BGR2GRAY);

	surf->detect(img, keypoints);
}

void FeatureExtractor::DetectHoNC(const Mat &img, vector<KeyPoint> &keypoints)
{
	(*honc)(img, noArray(), keypoints, noArray(), false);
}



void FeatureExtractor::ComputeSIFT(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	Mat greyImg;
	cvtColor(img, greyImg, COLOR_BGR2GRAY);

	sift->compute(greyImg, keypoints, descriptors);
}

void FeatureExtractor::ComputeSURF(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	Mat greyImg;
	cvtColor(img, greyImg, COLOR_BGR2GRAY);

	surf->compute(greyImg, keypoints, descriptors);
}

void FeatureExtractor::ComputeHoNC(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors)
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