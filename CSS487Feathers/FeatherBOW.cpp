#include "FeatherBOW.h"



FeatherBOW::FeatherBOW(ExtractType eType, int numWords, string name)
{
	extrType = eType;
	this->numWords = numWords;
	this->name = name;
}


FeatherBOW::~FeatherBOW()
{
}

void FeatherBOW::Train(const vector<Mat> &inputImages)
{

	//<for each entry in the training data>
	//1. run the data through the feature extractor
	//2. put the features into the BOW Trainer
	//3. Make the BOW histrogram
	//4. Add it to the SVM
}

void FeatherBOW::LoadData()
{

}

void FeatherBOW::SaveData()
{

}

float FeatherBOW::Predict(const Mat &input)
{
	//1. run the input throught the feature extractor
	//2. put the features through the BOW machine
	//3. SVM.predict()
	return 0.0f;
}
