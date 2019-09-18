import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import re 	# import regex to use when splitting
from collections import Counter

# COMMENT ATTRIBUTES
# The "data" variable is a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 
# lowercase_text : the "text" attributed converted to lowercase (can be removed if the "words" attribute is adequate)
# words : the words of the "lowercase_text" attribute in lowercase, split into elements of a list
# x_counts : holds at position i the number of times that the word at most_common_words[i][0] occurs in this comment

most_common_words = []	# a dictionary containing the 160 most common words in the data set, as well as the number of times they appear
						# sorted in descending order of number of occurences

# pre-processing function to gather information from incoming data set
def preprocess(comments):
	allWords = []																			# variable to hold overall list of all words
	for comment in comments:																# iterate through each of the supplied comments
		comment['lowercase_text'] = comment['text'].lower()									# convert all text to lowercase
		comment['words'] = re.findall(r"[\w']+|[.,!?;]", comment['lowercase_text'])			# split text into individual words
		allWords.extend(comment['words'])													# add to overall list of words
	
	# count most common words in the comments
	counter = Counter(allWords)
	most_common_words = counter.most_common(160)

	# determine x_counts for each of the comments
	for comment in comments:
		i = 0
		comment['x_counts'] = [0]*160
		for entry in most_common_words:
			comment_counter = Counter(comment['words'])
			comment['x_counts'][i] = comment_counter[entry[0]]
			i = i + 1

with open("proj1_data.json") as fp:			# load data from file
    data = json.load(fp)
    
preprocess(data)	# call the preprocessing function

# Example:
data_point = data[1] # select the first data point in the dataset

# Now we print all the information about this datapoint
for info_name, info_value in data_point.items():
    print(info_name + " : " + str(info_value))