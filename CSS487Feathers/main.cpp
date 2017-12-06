#include "FeatherIdentifier.h"

#include <iostream>
using namespace std;

#include <opencv2\core.hpp>
using namespace cv;

int main(int argc, char *argv[])
{
	string mode;
	string file1, file2, workingDirectory;

	if (argc < 4)
		return -1;

	//need to at least have a testing and training file
	if (argc >= 4)
	{
		mode = argv[1];
		file1 = argv[2];		//these are used differently depending upon context
		file2 = argv[3];
	}
	
	//optional working directory argument
	if (argc >= 5)
	{
		workingDirectory = argv[4];
		workingDirectory += "/";
	}
	else
	{
		workingDirectory = "";
	}

	FeatherIdentifier FeatherID = FeatherIdentifier(workingDirectory);
	bool verify = true;

	
	if (mode == "train+save")		//file1 = trainingFile, file2 = classifier
	{
		if (!FeatherID.Train(file1, verify, false))
		{
			cerr << "Failed to train Identifier!" << endl;
			system("pause");
			return -1;
		}

		if (!FeatherID.Save(file2))
		{
			cerr << "Failed to save Identifier!" << endl;
			system("pause");
			return -1;
		}
	}
	else if (mode == "load+test")	//file1 = classifier, file2 = testing data
	{
		if (!FeatherID.Load(file1))
		{
			cerr << "Failed to load Identifier!" << endl;
			system("pause");
			return -1;
		}

		if (!FeatherID.Identify(file2, false, false))
		{
			cerr << "Identification Failed!" << endl;
			system("pause");
			return -1;
		}

	}
	else if (mode == "train+test")	//file1 = training data, file2 = testing data
	{

		if (!FeatherID.Train(file1, verify, false))
		{
			cerr << "Failed to train Identifier!" << endl;
			system("pause");
			return -1;
		}


		if (!FeatherID.Identify(file2, false, false))
		{
			cerr << "Identification Failed!" << endl;
			system("pause");
			return -1;
		}
	}
	else
	{
		cerr << "invalid mode!" << endl;
		cerr << "you put \"" << mode << "\"" << endl;
		system("pause");
		return -1;
	}

	system("pause");
	return 0;
}