# -*- coding: utf-8 -*-
# __author__: Yixuan LI
# __email__: yl2363@cornell.edu

import os
import json
import re
from optparse import OptionParser
import tweepy
import time

class UserTimeline:

	def __init__(self,inputDir,outputDir):

		self.inputDir = inputDir
		self.outputDir = outputDir
		os.system("mkdir -p %s"%(outputDir))

		# Get the names of the files under the input directory and save them in a list
		self.fileList = os.listdir(inputDir)
		print self.fileList
		self.userHash = {}			 # [key,value] pair to record the unique users in the tweets
		self.uniqueUserCount = 0     # count unique users in the dataset
		self.tweetCount = 0          # total tweets processed
		self.api = None

	def authentication(self):

		consumer_key="z86C8djY3bYOPD1WkYV73nVP6"
		consumer_secret="BT8oKrcj955MKjv0qS8Kra2Iw91E3uSMTqEVurfTmKjXfG0hNm"

		access_token="746349096-Bz1n8T6vNEFBAMG2YqVdJFOtrM321d5HeupxMlxM"
		access_token_secret="ZZQZsjvJXnIlyl04Mg2vCxS8g122b3AljpiytiKCKRFPL"
	
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)
		self.api = tweepy.API(auth)
		print "authentication finished"

	def get_user_id(self):
		written = 0
		if os.path.exists(self.outputDir + "/" + "uniqueUserID.txt"):
			pass
		else:
			for tweetFile in self.fileList[1:]:
				with open(self.inputDir+"/"+tweetFile,'r') as fin:
					for line in fin:
						try:
							lineContents = json.loads(line)   # load a line
							self.tweetCount += 1
							print self.tweetCount             # for debugging
						except:
							continue
						try:	
							if lineContents["coordinates"] is not None:
								continue
							else:
								# extract user's id 
								userID = lineContents["user"]["id"]
								# extract tweet text and convert the string to lower case (http://stackoverflow.com/questions/6797984/how-to-convert-string-to-lowercase-in-python)
								#tweet = lineContents["text"].lower()

								if not self.userHash.has_key(userID):  # if the user has not been counted 
									self.uniqueUserCount += 1                # count the number of unique users
									self.userHash[userID] = True
								
									fileNum = int(self.uniqueUserCount/7250 + 1)
									with open(self.outputDir + "/" + "uniqueUserID_"+str(fileNum)+".txt","a") as fileout:
										written += 1
										fileout.write(str(userID))
										fileout.write("\n")
										print written," written"
						except:
							continue

			print "There are ", self.uniqueUserCount, "unique users"
			print self.tweetCount, " tweets processed"

	def get_user_timeline(self):

		
		with open(self.outputDir + "/" + "uniqueUserID_6.txt",'r') as fin:
			for userID in fin:	
				# store the tweets of each user in a single file named by the {userID}.json
				filePath = self.outputDir + "/" + str(userID[:-1])+".json"
				print userID
				if os.path.exists(filePath):
					with open(filePath,'r') as myfile:
								count = sum(1 for line in myfile)
								if count > 900:
									continue
								else:
									# http://stackoverflow.com/questions/6996603/how-do-i-delete-a-file-or-folder-in-python
									os.remove(filePath)

				pageCount = 1
				trialTime = 0
				# get user timeline tweets
				while pageCount < 6:
					print "Collecting", pageCount, " -th page"
					# open the output file in append mode		
					self.fout = open(filePath,"a")
					try:
						tweets = self.api.user_timeline(id=userID,count=200,page=pageCount)
						pageCount += 1
					except:
						time.sleep(70)
						trialTime += 1
						if trialTime == 2:
							pageCount = 8    
							continue
					# write to file
					# Note that data returned by api.user_timeline is status object
					for tweet in tweets:
						print tweet.text
						# convert tweepy status object to json format
						# http://stackoverflow.com/questions/27900451/convert-tweepy-status-object-into-json
						self.fout.write(json.dumps(tweet._json))
						self.fout.write('\n')
					time.sleep(70)            # rate limit (15 requests per 15 minutes window)



if __name__=='__main__':


#########################################################################################
    # Parse the arguments
    class MyParser(OptionParser):
        def format_epilog(self, formatter):
            return self.epilog
    
    usage = "usage: python plot_stats.py [options]"
    description = """
    """
    epilog = """

    """
    parser = MyParser(usage, description=description,epilog=epilog)
    parser.add_option("--inputDir", "--input file of twitter data", dest="input_path", default=None,
                      help="input directory of twitter streaming data in JSON format [default: None]")
    parser.add_option("--outputDir", "--output directory of twitter user  timeline data", dest="output_path", default=None,
                      help="output directory of twitter user timeline data [default: None]")
    (options, args) = parser.parse_args()
    # input directory
    inputDir = options.input_path
    # output directory
    outputDir = options.output_path

########################################################################
    

    getter = UserTimeline(inputDir,outputDir)
    getter.authentication()
    #getter.get_user_id()
    getter.get_user_timeline()


					



