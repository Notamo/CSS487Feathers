#include "FeatureExtractor.h"


/*
FeatureExtractor() - Constructor
Precondition: No FeatureExtractor object
Postcondition: FeatureExtractor is created, and
	sift and surf member objects are initialized
*/
FeatureExtractor::FeatureExtractor()
{
	sift = SIFT::create();
	surf = SURF::create();
}


FeatureExtractor::~FeatureExtractor()
{
}

/*
GetFD - Get a FeatureDetector
Precondition: caller provides an ExtractType type 
	and a reference to a Ptr<FeatureDetector> &FD
Postcondition: returns success state, and FD is set
	to the FeatureDetector indicated by type
*/
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

/*
GetDE - Get a DescriptorExtractor
Precondition: caller provides an ExtractType type
	and a reference to a Ptr<FeatureDetector> &FD
Postcondition: returns success state, and FD is set
	to the DescriptorExtractor indicated by type
*/
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

/*
StringToExtractType
Precondition: caller provides a string and a reference
	to an ExtractType &type
Postcondition: returns success state, type is set to
	the extract type indicated by str
*/
bool StringToExtractType(const string &str, ExtractType &type)
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

/*
ExtractTypeToString
Precondition: caller provides the extractor type
	desired
Postcondition: returns the ExtractType indicated
	as a string
*/
string ExtractTypeToString(ExtractType type)
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