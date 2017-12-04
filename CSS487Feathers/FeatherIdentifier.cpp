#include "FeatherIdentifier.h"



FeatherIdentifier::FeatherIdentifier(const string &workingDirectory)
{
	this->workingDirectory = workingDirectory;
}


FeatherIdentifier::~FeatherIdentifier()
{
}

bool FeatherIdentifier::TrainIdentifier(const string &trainingFile)
{
	ExtractType eType = ExtractType::E_None;
	int numWords = -1;

	if (!MakeTrainingSets(trainingFile, eType, numWords))
	{
		cerr << "Failed to make training sets!" << endl;
		return false;
	}

	if (!TrainBOWs(eType, numWords))
	{
		cerr << "Failed to train BOWs!" << endl;
		return false;
	}

	if (!TrainSVMs(eType, numWords))
	{
		cerr << "Failed to train SVM!" << endl;
		return false;
	}

	trained = true;
	return true;
}

bool FeatherIdentifier::TrainBOWs(ExtractType eType, int numWords)
{

	if (trainingSets.empty())
	{
		cerr << "training set empty!" << endl;
		return false;
	}

	if (numWords <= 0)
	{
		cerr << "too few words! (" << numWords << ")" << endl;
		return false;
	}

	BOW = new FeatherBOW(eType, numWords);
	BOW->MakeDictionary(trainingSets);
	return true;
}

bool FeatherIdentifier::TrainSVMs(ExtractType eType, int numWords)
{
	Mat samples;
	Mat outSamples;
	Mat outLabels;

	outSamples.create(0, 1, CV_32FC1);
	outLabels.create(0, 1, CV_32SC1);

	for (int index = 0; index < trainingSets.size(); index++)
	{
		for (int im = 0; im < trainingSets[index].images.size(); im++)
		{
			vector<KeyPoint> keypoints;

			Mat descriptors;
			if (!BOW->ComputeImgHist(trainingSets[index].images[im], descriptors))
			{
				cerr << "Cannot train SVM! Unable to compute histogram" << endl;
				return false;
			}

			if (!descriptors.empty())
			{
				if (samples.empty())
				{
					samples.create(0, descriptors.cols, descriptors.type());
				}

				cout << "Adding " << descriptors.rows << "samples for category: " << index << endl;
				samples.push_back(descriptors);

				Mat classLabels;

				classLabels = Mat(descriptors.rows, 1, CV_32SC1);
				
				//fill each with the class
				for(int i = 0; i < classLabels.cols; i++)
					for(int j = 0; j < classLabels.rows; j++)
						classLabels.at<int>(i, j) = index;

				outLabels.push_back(classLabels);
			}
			else
			{
				cout << "no descriptors!" << endl;
			}

		}

		if (samples.empty() || outLabels.empty())
		{
			cout << "samples is empty" << endl;
		}

		samples.convertTo(outSamples, CV_32FC1);
	}

	singleSVM = SVM::create();
	singleSVM->train(outSamples, ROW_SAMPLE, outLabels);

	return true;
}


//Train One individual SVM
//When training, add the correct categories with label values = 1
//and add ALL the other categories with label values = 0
bool FeatherIdentifier::TrainSVM(int index, int numWords)
{
	return true;
}

//basic idea:
//look for the image's response histogram to the vocabulary of features
//run it by all classifiers, and pick the one with the best score
bool FeatherIdentifier::Identify(const string &testFile, vector<RatingPair> &ratings)
{
	if (!trained)
	{
		cerr << "Feather Identifier not trained!" << endl;
		return false;
	}

	Mat testImg = imread(testFile);
	Mat responseHist;

	if (testImg.data == nullptr)
	{
		cerr << "Failed to open file!" << endl;
		return false;
	}
	
	//get the histogram for the input image
	if (!BOW->ComputeImgHist(testImg, responseHist))
	{
		cerr << "Cannot compute histogram for test image!" << endl;
		return false;
	}

	if (responseHist.empty())
	{
		cerr << "Empty response histogram!" << endl;
		return false;
	}

	/*for (int i = 0; i < SVMs.size(); i++)
	{
		Mat results;
		float response = SVMs[i]->predict(responseHist, results);

		cout << "category: " << trainingSets[i].name << endl;
		cout << "score: " << response << endl;


		for (int i = 0; i<results.rows; i++)
		for (int j = 0; j<results.cols; j++)
		printf("results(%d, %d) = %d \n", i, j, results.at<float>(i, j));

	}*/

	Mat results;
	float res = singleSVM->predict(responseHist);
	cout << "prediction result: " << res << endl;

	//Build a vector of RatingPairs, 
	//based on the results of calling FeatherBOW.Predict
	/*for (int i = 0; i < BOWs.size(); i++)
	{
		float rating = Predict(BOWs[i], testImg);
		RatingPair rnPair;
		rnPair.name = BOWs[i]->GetName();
		rnPair.rating = rating;
		ratings.push_back(rnPair);
	}*/

	return true;
}


bool FeatherIdentifier::SaveBOWs(const string &bowFile)
{
	//stretch goal
	return false;
}

bool FeatherIdentifier::LoadBOWs(const string &bowFile)
{
	//stretch goal
	return false;
}



void FeatherIdentifier::ListResults(const vector<RatingPair> &pairs)
{
	//1. Sort the pairs

	//2. print the pairs out
	for (int i = 0; i < pairs.size(); i++)
	{
		cout << pairs[i].name << ": " << pairs[i].rating << endl;;
	}
}



//Read in a file <dbFile> that describes the name of each set,
//POTENTIAL FORMAT:
//<Extraction method>	<# Words>
//<set name>	<set size>	<directory>	
//<set name>	<set size>	<directory>
bool FeatherIdentifier::MakeTrainingSets(const string &trainingFile, ExtractType &eType, int &numWords)
{
	//try to open the trainingFile
	ifstream TF(workingDirectory + trainingFile);

	if (!TF.is_open())
	{
		cerr << "Failed to open trainingFile! (" << workingDirectory + trainingFile << ")" << endl;
		return false;
	}

	//read the first line, should be in the following format\:
	//<SIFT | SURF | HoNC>	<TAB>	<# words>
	string line;
	stringstream ss;
	if (getline(TF, line))
	{
		string methodStr;

		ss << line;
		ss >> methodStr >> numWords;

		//convert the methodString into
		eType = ExtractTypeFromString(methodStr);

		if (eType == ExtractType::E_None)
		{
			cerr << "Invalid extract type!" << endl;
			TF.close();
			return false;
		}

		if (numWords < MIN_FEATHER_WORDS)
		{
			cerr << "Invalid number of words!" << endl;
			TF.close();
			return false;
		}
	}
	else
	{
		TF.close();
		return false;
	}


	while (getline(TF, line))
	{
		TrainingSet TS;
		string name;
		int qty;
		string subdir;

		ss.clear();
		ss << line;
		ss >> name >> qty >> subdir;

		
		if (!BuildTrainingSet(subdir, name, qty, TS))
		{
			cerr << "Failed to build training set for \"" << name << "\"" << endl;
			TF.close();
			return false;
		}

		trainingSets.push_back(TS);
	}

	TF.close();
	return true;
}

bool FeatherIdentifier::BuildTrainingSet(const string &subdir, const string &name, const int &qty, TrainingSet &set)
{
	cout << "Building Set: (" << name << " " << qty << " " << subdir << ")" << endl;
	set.name = name;

	for (int i = 0; i < qty; i++)
	{
		//string file(workingDirectory + subdir + name + "_" + to_string(i) + ".jpg");
		string file(workingDirectory + subdir + name + " (" + to_string(i) + ").jpg");


		cout << "loading image \"" << file << "\"" << endl;
		Mat m = imread(file);

		if (m.data == nullptr)
		{
			cerr << "Could not open \"" << file << "\"!" << endl;
			cerr << "Failed to build " << name << " set!" << endl;
			return false;
		}

		set.images.push_back(m);
	}
	return true;
}