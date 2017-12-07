#include "BOWDataMgmt.h"



/*
MakeTrainingSets
Precondition: workingDirectory corresponds to a valid directory
	arguments:
		workingDirectory - the directory all file IO is relative to
Postcondition: this->workingDirectory is set to the input working directory
*/
BOWDataMgmt::BOWDataMgmt(const string &workingDirectory)
{
	this->workingDirectory = workingDirectory;
}


BOWDataMgmt::~BOWDataMgmt()
{
}


/*
MakeTrainingSets
Precondition: trainingFile corresponds to a valid training file, and trainingSets is empty
	arguments:
		trainingFile - the file that indicates training variables and
		trainingSets - The trainingSets to fill out
		eType - the ExtracType to assign
		numWords - the variable to assign the number of words
Postcondition: trainingSets is filled with sets of images to train the Identifier with. eType
	and numWords are set their new values
	returns:
		success state (T/F)
*/
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
		if (!StringToExtractType(methodStr, eType))
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

/*
MakeTrainingSets
Precondition: testingFile corresponds to a valid testing file, and testingSets is empty
	arguments:
		testingFile - the file that indicates where to access the training images
		trainingSets - The trainingSets to fill out
Postcondition: testingSets is filled with sets of images to test the Identifier with.
	returns:
	success state (T/F)
*/
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

	string line;
	stringstream ss;

	int label = 0;
	while (getline(TF, line))
	{
		if (line[0] == '#')			//lines starting with '#' are comments
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

/*
BuildImageSet
Precondition: subdir corresponds to an existing subdirectory, name coresponds to a valid
	category name (you can find images of the following naming scheme: [name]_#)
	arguments:
		subdir - the subdirectory where the images reside
		name - the name of the set of images
		label - the label of the current set of images
		qty - the number of images to add to the set
		set - the ImageSet to build
Postcondition: set is filled with the images of the desired category
	returns:
		success state (T/F)
*/
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

/*
SaveSVM - NOT YET IMPLEMENTED
Precondition: workingDirectory corresponds to an existing subdirectory
	arguments:
		fileName - the name of the file to save the svm data in
		svm - theSVM containing the data to save
Postcondition: the SVM data is stored in a file of fileName in workingDirectory/saveDir/
	returns:
		success state (T/F)
*/
bool BOWDataMgmt::SaveSVM(const string &fileName, Ptr<SVM> &svm)
{
	svm->save(workingDirectory + saveDir + fileName);					//is there a way to check if this succeeded?
	return true;
}

/*
LoadSVM - NOT YET IMPLEMENTED
Precondition: workingDirectory corresponds to an existing subdirectory, fileName corresponds
	to a file that exists
	arguments:
		fileName - the name of the file to load
		svm - theSVM to load data into
Postcondition: the SVM is loaded with the data in the file "workingDirectory/saveDir/fileName"
	returns:
		success state (T/F)
*/
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

/*
SaveDictionary - NOT YET IMPLEMENTED
Precondition: workingDirectory corresponds to an existing subdirectory
	arguments:
		fileName - the name of the file to save the dictionary in
		dictionary - the dictionary to save
Postcondition: the dictionary is stored in the file "workingDirectory/saveDir/fileName"
	returns:
		success state (T/F)
*/
bool BOWDataMgmt::SaveDictionary(const string &fileName, const Mat &dictionary)
{
	return imwrite(workingDirectory + saveDir + fileName, dictionary);
}

/*
LoadDictionary - NOT YET IMPLEMENTED
Precondition: workingDirectory corresponds to an existing subdirectory, fileName corresponds
	to a file that exists
	arguments:
		fileName - the name of the file to load from
		svm - the dictionary to load
Postcondition: the SVM is loaded with the data in the file "workingDirectory/saveDir/fileName"
	returns:
		success state (T/F)
*/
bool BOWDataMgmt::LoadDictionary(const string &fileName, Mat &dictionary)
{
	imread(workingDirectory + saveDir + fileName);

	return (dictionary.data != nullptr);
}