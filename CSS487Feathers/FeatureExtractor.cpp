#include "FeatureExtractor.h"


FeatureExtractor::FeatureExtractor()
{
	sift = SIFT::create();
	surf = SURF::create();
}


FeatureExtractor::~FeatureExtractor()
{
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
	else
		return false;

	return true;
}

string StringFromExtractType(ExtractType type)
{
	switch (type)
	{
	case ExtractType::E_SIFT:
		return "SIFT";
	case ExtractType::E_SURF:
		return "SURF";
	case ExtractType::E_None:
		return "NONE";
	default:
		return "ERROR";
	}
}