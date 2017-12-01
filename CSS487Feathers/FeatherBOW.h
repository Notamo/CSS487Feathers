#pragma once
#include <opencv2\features2d.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace ml;

#include <string>
#include <vector>
using namespace std;

#include "FeatureExtractor.h"

const int MIN_FEATHER_WORDS = 10;

/*
FeatherBOW
Description:
This class Represents a Bag of Words (BOW) for a specific feather
The BOW can either be trained from a Data Set, or the already trained
Data can be loaded in from a file. Once trained data is in the BOW,
An input can be compared to the BOW to determine how likely it is in the set
*/
class FeatherBOW
{
public:
	FeatherBOW(ExtractType eType, int numWords, string name);
	~FeatherBOW();

	//Creates the BOW, by training a set of Images
	void MakeDictionary(const vector<Mat> &inputImages);
	Mat GetDictionary();
	int GetSize();

	string GetName() { return name; }

	bool ComputeImgHist(const Mat &descriptors, Mat &hist);

	//STRETCH GOAL: load and save training state for faster detection
	void LoadData(string directory);
	void SaveData(string directory);

private:
	FeatureExtractor extractor;
	ExtractType extrType = ExtractType::E_None;
	int numWords = 10;
	string name = "[no name]";
	bool trained = false;

	Mat dictionary;
	
	//for generating a bag of words/histogram (one for each training image)
	Ptr<BOWKMeansTrainer> bowTrainer;
	int numAttempts = 3;

	

	
};

