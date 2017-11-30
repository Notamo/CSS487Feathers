#pragma once
#include <opencv2\features2d.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace ml;

#include <string>
#include <vector>
using namespace std;

#include "FeatureExtractor.h"
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
	void Train(const vector<Mat> &inputImages);

	//Tells the user how likely it is the input image falls under this category
	float Predict(const Mat &input);

	//STRETCH GOAL: load and save training state for faster detection
	void LoadData();
	void SaveData();
private:
	ExtractType extrType = ExtractType::None;
	
	//for generating a bag of words/histogram
	Ptr<BOWKMeansTrainer> bowTrainer;
	int numClusters = 10;	//number of words in vocab
	int numAttempts = 3;

	//Support Vector Machine actually generates a way to predict
	//using our GIANT histogram
	Ptr<SVM> svm;

	
};

