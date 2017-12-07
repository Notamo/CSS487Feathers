import os

workingDirectory = "tigers/"
extension = ".jpg"
old_category = "3"
new_category = "tiger"

for x in range (1, 64):
	old_file = workingDirectory + old_category + " (" + str(x) + ")" + extension
        #old_file = workingDirectory + old_category + "_" + str(x) + extension
	new_file = workingDirectory + new_category + "_" + str(x - 1) + extension
	print "renaming \"" + old_file + "\" to \"" + new_file + "\""
	os.rename(old_file, new_file)
