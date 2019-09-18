import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import re 	# import regex to use when splitting
from collections import Counter

most_common_words = []

# pre-processing function to gather information from incoming data set
def preprocess(dictionaries):
	allWords = []																			# variable to hold overall list of all words
	for dictionary in dictionaries:															# iterate through each of the supplied comments
		dictionary['lowercase_text'] = dictionary['text'].lower()							# convert all text to lowercase
		dictionary['words'] = re.findall(r"[\w']+|[.,!?;]", dictionary['lowercase_text'])	# split text into individual words
		allWords.extend(dictionary['words'])												# add to overall list of words
	
	# count most common words in the comments
	counter = Counter(allWords)
	most_common_words = counter.most_common(160)
	print(most_common_words)						# TODO remove

	# # determine x_counts for each of the comments
	# for dictionary in dictionaries:
	# 	i = 0
	# 	dictionary['x_counts'] = [0] * 160
	# 	for entry in most_common_words:
	# 		print('entry : ' + entry[0])
	# 		single_comment_counter = Counter(dictionary['words'])
	# 		dictionary['x_counts'][i] = single_comment_counter[entry]
	# 		i = i + 1

with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

preprocess(data)	# call the preprocessing function

# Example:
data_point = data[1] # select the first data point in the dataset

# Now we print all the information about this datapoint
for info_name, info_value in data_point.items():
    print(info_name + " : " + str(info_value))