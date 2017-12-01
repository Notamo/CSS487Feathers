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
	vector<TrainingSet> trainingSets;
	ExtractType eType = ExtractType::E_None;
	int numWords = -1;

	if (!MakeTrainingSets(trainingFile, trainingSets, eType, numWords))
	{
		cerr << "Failed to make training sets!" << endl;
		return false;
	}

	if (!TrainBOWs(trainingSets, eType, numWords))
	{
		cerr << "Failed to train BOWs!" << endl;
		return false;
	}

	trained = true;
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

	//Build a vector of RatingPairs, 
	//based on the results of calling FeatherBOW.Predict
	for (int i = 0; i < BOWs.size(); i++)
	{
		float rating = BOWs[i]->Predict(testImg);
		RatingPair rnPair;
		rnPair.name = BOWs[i]->GetName();
		rnPair.rating = rating;
		ratings.push_back(rnPair);
	}

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

bool FeatherIdentifier::TrainBOWs(const vector<TrainingSet> &trainingSets, ExtractType eType, int numWords)
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
		BOW->Train(trainingSets[i].images);
		BOWs.push_back(BOW);
	}

	return true;
}

//Read in a file <dbFile> that describes the name of each set,
//POTENTIAL FORMAT:
//<Extraction method>	<# Words>
//<set name>	<set size>	<directory>	
//<set name>	<set size>	<directory>
bool FeatherIdentifier::MakeTrainingSets(const string &trainingFile, vector<TrainingSet> &trainingSets, ExtractType &eType, int &numWords)
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