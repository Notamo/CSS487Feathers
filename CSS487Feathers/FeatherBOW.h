#pragma once
#include <opencv2\features2d.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace ml;

#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include "FeatherIDUtil.h"
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
	FeatherBOW(ExtractType eType, int numWords);
	~FeatherBOW();

	//Creates the BOW, by training a set of Images
	void MakeDictionary(const vector<TrainingSet> &trainingSets);
	Mat GetDictionary();
	int GetSize();

	bool ComputeImgHist(const Mat &image, Mat &hist);

	//STRETCH GOAL: load and save training state for faster detection
	void LoadDictionary(string directory);
	void SaveDictionary(string directory);

private:
	FeatureExtractor extr;
	ExtractType extrType = ExtractType::E_None;

	int numWords = 10;
	bool trained = false;

	Mat dictionary;

	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<DescriptorMatcher> matcher;
	Ptr<BOWKMeansTrainer> bowTrainer;
	Ptr<BOWImgDescriptorExtractor> bowDE;
	int numAttempts = 3;

	

	
};

