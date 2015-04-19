# -*- coding: utf-8 -*-
# __author__: Yixuan LI
# __email__: yl2363@cornell.edu


import os
import json
import re
from optparse import OptionParser

"""Process the raw twitter data (in JSON format).
1. Filter the tweets by un-frequent users whose life-time tweets are less than 400.
2. Filter the tweets without specified geo location or coordinates
3. Clean retweets
4. Filter non-English tweets
5. Filter truncated tweets
"""


class TweetsCleaning:

	def __init__(self,rawDir,outputDir):
		self.outputDir = outputDir
		self.rawDir = rawDir
		os.system("mkdir -p %s"%(outputDir))

		# extract the basename of the input data directory
		self.filename = os.path.basename(rawDir) 

		self.emptyGeoCount = 0        # count tweets with geo information
		self.tweetsCount = 0          # count total tweets
		self.rtCount = 0              # count retweets
		self.rareUserCount = 0        # count number of tweets by users with less than 400 life time tweets
		self.truncatedCount = 0       # count the number of truncated tweets
		self.sensitiveCount = 0       # count the number of sensitive tweets
		self.writtenCount = 0

	def cleanTweets(self,deleteRT,withGeo,deleteRareUser):
		
		# open the output file in append mode
		self.fout = open(self.outputDir + "/" + self.filename,"a")

		# open the raw data for reading only 
		with open(self.rawDir,'r') as fin:
			for line in fin:
				
				try:
					lineContents = json.loads(line)   # load a line
				except ValueError, e:
					pass
				self.tweetsCount += 1
				print self.tweetsCount
				# parse the json data. 
				# Details of the  tweet format can be found here: https://dev.twitter.com/overview/api/tweets
				
				if 'lang' not in lineContents:
					continue

				if lineContents["lang"] != "en":      # filter non-english tweets
					continue  
				
				if lineContents["truncated"] == True:      # filter truncated tweets
					continue 

				# check if it is retweet      
				elif re.search('RT*',lineContents["text"]) is not None:
					self.rtCount += 1
					if deleteRT: continue
				
				# check if it is tweeted by a unfrequent user
				elif int(lineContents["user"]["statuses_count"]) < 1000: 
					self.rareUserCount += 1
					if deleteRareUser: continue

				# filter sensitive tweets
				elif int(lineContents["possibly_sensitive"]) == True: 
					self.sensitiveCount += 1
					continue

				# check if it contains geo location and coordinatea
				elif lineContents["coordinates"] is None or lineContents["user"]["location"] == "":
					self.emptyGeoCount += 1
					if withGeo: continue

				self.fout.write(line)
				self.writtenCount += 1

		print str(self.rareUserCount) + " tweets by users with less than 1000 life time tweets"
		print str(self.emptyGeoCount) + " tweets do not contain geo information"
		print str(self.rtCount) + " retweets"
		print str(self.writtenCount) + " tweets after cleaning"
		print str(self.tweetsCount) + " tweets in total"


if __name__=='__main__':


#########################################################################################
    # Parse the arguments
    class MyParser(OptionParser):
        def format_epilog(self, formatter):
            return self.epilog
    
    usage = "usage: python filter_tweets.py [options]"
    description = """
    """
    epilog = """

    """
    parser = MyParser(usage, description=description,epilog=epilog)
    parser.add_option("--rawDir", "--input file of raw twitter streaming data", dest="input_path", default=None,
                      help="input file of unprocessed, raw twitter streaming data in JSON format [default: None]")

    parser.add_option("--outputDir", "--output directory of processed twitter data", dest="output_path", default=None,
                      help="output directory to save processed twitter streaming data (in JSON format) [default: None]")

    parser.add_option("--deleteRT", "--delete retweets", dest="deleteRT", default=True,
                      help="when set to be True, delete all the retweeted posts. [default: False]")

    parser.add_option("--withGeo", "--delete tweets without specified geographic location", dest="withGeo", default=True,
                      help="when set to be True, only tweets with user's geographic location specified are kept [default: True]")

    
    parser.add_option("--deleteRareUser", "--delete users with life-time tweets less than 100", dest="deleteRareUser", default=True,
                      help="when set to be True, delete all the users with life time tweets less than 100 [default: False]")


    (options, args) = parser.parse_args()
    # input directory
    rawDir = options.input_path
    # output directory, this should be different from the input directory
    outputDir = options.output_path
    # delete retweets? 
    deleteRT = options.deleteRT
    # delete tweets without geo information?
    withGeo = options.withGeo
    # delete all the users with less than 400 life time tweets?
    deleteRareUser = options.deleteRareUser

########################################################################
    

    cleaner = TweetsCleaning(rawDir,outputDir)
    cleaner.cleanTweets(deleteRT,withGeo,deleteRareUser)
    print "Finish processing!"



