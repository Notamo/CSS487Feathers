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

	if (!TrainSVM(eType, numWords))
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

	//get all
	for (int i = 0; i < trainingSets.size(); i++)
	{
		//make a new FeatherBOW
		Ptr<FeatherBOW> BOW = new FeatherBOW(eType, numWords, trainingSets[i].name);
		BOW->MakeDictionary(trainingSets[i].images);
		BOWs.push_back(BOW);
	}

	return true;
}

bool FeatherIdentifier::TrainSVM(ExtractType eType, int numWords)
{
	FeatureExtractor extractor;

	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, numWords, CV_32FC1);

	//for each category
	for (int i = 0; i < trainingSets.size(); i++)
	{
		//compute the histograms of each image's
		//relationship with the dictionary of the category
		for (int j = 0; j < trainingSets[i].images.size(); j++)
		{
			//For each image, make a histogram counting the instances of each word
			//in the dictionary. Then, add that histogram to the svm
			vector<KeyPoint> keypoints;
			Mat descriptors;
			Mat histogram;

			extractor.ExtractFeatures(eType, trainingSets[i].images[j], keypoints, descriptors);

			BOWs[i]->ComputeImgHist(descriptors, histogram);

			cout << "descriptors size: " << descriptors.size() << endl;
			cout << "hist size: " << histogram.size() << endl;

			//add to our data
			trainingData.push_back(histogram);
			labels.push_back(i);
		}
	}

	cout << trainingData.size() << endl;
	cout << labels.size() << endl;

	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	svm->trainAuto(trainingData, ROW_SAMPLE, labels);

	return true;
}

bool FeatherIdentifier::Identify(const string &testFile, vector<RatingPair> &ratings)
{
	if (!trained)
	{
		cerr << "Feather Identifier not trained!" << endl;
		return false;
	}

	Mat testImg = imread(testFile);

	if (testImg.data == nullptr)
	{
		cerr << "Failed to open file!" << endl;
		return false;
	}

	namedWindow("testImg");
	imshow("testImg", testImg);
	waitKey(0);

	vector<KeyPoint> keypoints;
	Mat descriptors;
	FeatureExtractor extractor;
	extractor.ExtractFeatures(eType, testImg, keypoints, descriptors);

	
	vector<Mat> histograms;
	for (int i = 0; i < trainingSets.size(); i++)
	{
		Mat newHist;
		BOWs[i]->ComputeImgHist(descriptors, newHist);
		histograms.push_back(newHist);
	}

	for (int i = 0; i < histograms.size(); i++)
	{
		float response = svm->predict(histograms[i]);
		cout << "closest guess: " << response << endl;
	}



	
	
	namedWindow("testImg");
	imshow("testImg", testImg);
	waitKey(0);

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
		string file(workingDirectory + subdir + name + "_" + to_string(i) + ".jpg");

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