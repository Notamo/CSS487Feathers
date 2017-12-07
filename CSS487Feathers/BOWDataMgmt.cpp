#include "BOWDataMgmt.h"



BOWDataMgmt::BOWDataMgmt(const string &workingDirectory)
{
	this->workingDirectory = workingDirectory;
}


BOWDataMgmt::~BOWDataMgmt()
{
}

//Read in a file <dbFile> that describes the name of each set,
//FORMAT:
//<Extraction method>	<# Words>
//<set name>	<set size>	<directory>	
//<set name>	<set size>	<directory>
bool BOWDataMgmt::MakeTrainingSets(const string &trainingFile, vector<ImageSet> &trainingSets, ExtractType &eType, int &numWords)
{
	cout << "Opening training File \"" << workingDirectory + trainingFile << "\"" << endl;
	//try to open the trainingFile
	ifstream TF(workingDirectory + trainingFile);
	if (!TF.is_open())
	{
		cerr << "Failed to open trainingFile! (" << workingDirectory + trainingFile << ")" << endl;
		return false;
	}

	//read the first line, should be in the following format\:
	//<SIFT | SURF>	<TAB>	<# words>
	string line;
	stringstream ss;
	if (getline(TF, line))
	{
		string methodStr;

		ss << line;
		ss >> methodStr >> numWords;

		//convert the methodString into
		if (!ExtractTypeFromString(methodStr, eType))
		{
			cerr << "Invalid Extract type!" << endl;
			return false;
		}

		if (numWords < MIN_FEATHER_WORDS)
		{
			cerr << "Invalid number of words!" << endl;
			TF.close();
			return false;
		}

		cout << "Training Set: " << methodStr << ", " << numWords << " words" << endl;
	}
	else
	{
		TF.close();
		return false;
	}



	int label = 0;
	while (getline(TF, line))
	{
		if (line[0] == '#')			//comment
		{
			continue;
		}

		ImageSet TS;
		string name;
		int qty;
		string subdir;

		ss.clear();
		ss << line;
		ss >> name >> qty >> subdir;

		if (!BuildImageSet(subdir, name, label, qty, TS))
		{
			cerr << "Failed to build training set for \"" << name << "\"" << endl;
			TF.close();
			return false;
		}

		trainingSets.push_back(TS);
		label++;
	}

	TF.close();
	return true;
}

bool BOWDataMgmt::MakeTestingSets(const string &testingFile, vector<ImageSet> &testingSets)
{
	cout << "Opening testing File \"" << workingDirectory + testingFile << "\"" << endl;

	//try to open the testingFile
	ifstream TF(workingDirectory + testingFile);

	if (!TF.is_open())
	{
		cerr << "Failed to open testingFile! (" << workingDirectory + testingFile << ")" << endl;
		return false;
	}

	//read the first line, should just have the extraction method
	//<SIFT>
	string line;
	stringstream ss;

	int label = 0;
	while (getline(TF, line))
	{
		if (line[0] == '#')			//comment
			continue;

		ImageSet TS;
		string name;
		int qty;
		string subdir;

		ss.clear();
		ss << line;
		ss >> name >> qty >> subdir;

		if (!BuildImageSet(subdir, name, label, qty, TS))
		{
			cerr << "Failed to build testing set for \"" << name << "\"" << endl;
			TF.close();
			return false;
		}

		testingSets.push_back(TS);
		label++;
	}

	TF.close();
	return true;
}

bool BOWDataMgmt::BuildImageSet(const string &subdir, const string &name, const int &label, const int &qty, ImageSet &set)
{
	cout << "Building Set: (" << name << " " << qty << " " << subdir << ")...";;
	set.name = name;
	set.label = label;

	for (int i = 0; i < qty; i++)
	{
		string file(workingDirectory + subdir + name + "_" + to_string(i) + ".jpg");
		Mat m = imread(file);

		if (m.data == nullptr)
		{
			cerr << "Could not open \"" << file << "\"!" << endl;
			cerr << "Failed to build " << name << " set!" << endl;
			return false;
		}

		set.images.push_back(m);
	}

	cout << "Done!" << endl;
	return true;
}

bool BOWDataMgmt::SaveSVM(const string &fileName, Ptr<SVM> &svm)
{
	svm->save(workingDirectory + saveDir + fileName);					//is there a way to check if this succeeded?
	return true;
}

bool BOWDataMgmt::LoadSVM(const string &fileName, Ptr<SVM> &svm)
{
	svm = SVM::load(workingDirectory + saveDir + fileName);

	if (svm == nullptr)
	{
		cerr << "Failed to load classifier! (" << (workingDirectory + fileName) << endl;
		return false;
	}

	return true;
}

bool BOWDataMgmt::SaveDictionary(const string &fileName, const Mat &dictionary)
{
	return imwrite(workingDirectory + saveDir + fileName, dictionary);
}

bool BOWDataMgmt::LoadDictionary(const string &fileName, Mat &dictionary)
{
	imread(workingDirectory + saveDir + fileName);

	return (dictionary.data != nullptr);
}