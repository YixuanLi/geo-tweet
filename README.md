# geo-tweet

Details of the project report can be found [here](http://www.cs.cornell.edu/~yli/projects/INFO6010/geo-tweet-analysis.html)


Requirements
------------
* Python >= 2.6 (but not 3.x)
* numpy
* sklearn ([http://scikit-learn.org/stable/install.html](http://scikit-learn.org/stable/install.html))
* tweepy ([https://github.com/tweepy/tweepy](https://github.com/tweepy/tweepy))
* twitter-python ([https://github.com/computermacgyver/twitter-python](https://github.com/computermacgyver/twitter-python))
* matplotlib ([http://matplotlib.org/users/installing.html] (http://matplotlib.org/users/installing.html))

(Note: the scripts of "twitter-python" have been included in this package under the directory of ``twitter-collect``, you don’t have to manually install by yourself. The author has made some slight changes based on the original source code in order to solve the connection break problem during the streaming.)

Usage
-----

######Step1: Collect raw twitter data using streaming API (optional)######

- To start collecting data using twitter's API, you need a Twitter account and a Twitter application. Details of setting up your Twitter application can be found here: [https://github.com/computermacgyver/twitter-python](https://github.com/computermacgyver/twitter-python).

- Once you have your consumer key, consumer secret, access token and access token secret, open the file ``twitter-collect/auth.py`` and update the following lines with the information obtained in step 1:

```
  consumer_key="..your key..."
  consumer_secret="..your secret..."
  access_token="...your token..."
  access_token_secret="...your token secret..."
```

- Start streaming data with the following command:

```
  $cd twitter-collect
  $python streaming.py 
```
 
- The streamed data will be saved under the directory of ``twitter-collect/collected_tweets/{current date}.json``.
	
- You could actually skip the step of collecting data and jump directly to the next section of cleaning twitter raw data. For this demo, I've already included a small sample raw data of 1018 tweets streamed from Twitter on Jan. 30th. It can be found under the directory ``twitter-collect/collected_tweets/example.json``.


######Step 2: Clean the raw twitter data######

- change your directory to ``twitter-clean``
- Example usage: 
```
python filter_tweets.py --rawDir ../twitter-collect/collected_tweets/example.json --outputDir ../tweets/
```
- Command options: you could type ``python filter_tweets.py --help`` for detailed information of input parameters.

  **--rawDir**: input file of unprocessed, raw twitter streaming data in JSON format

  **--outputDir**: output directory to save processed twitter streaming data (in JSON format) 
	
	**--deleteRT**: when set to be True, delete all the retweeted posts. [default: True]
	
	**--withGeo**: when set to be True, only tweets with user's geographic location and tweets coordination specified are kept [default: True]
	
	**--deleteRareUser**: when set to be True, delete all the users with life time tweets less than 100 [default: True]
	
  e.g., if you want to keep all the retweeted posts, you should run:
```
python filter_tweets.py --rawDir ../twitter-collect/collected_tweets/example.json --outputDir ../tweets/ --deleteRT False
```

 - The processed tweets will be stored in the specified output directory ``tweets/example.json``. You should be able to see 4 tweets kept after the cleaning if you are using the demo data ``example.jason``.


######Step 3: Collect the timeline tweets of users######

- Change your directory to ``twitter-timeline``
- ``python get_user_timeline.py --inputDir ../tweets/example.json --outputDir ../timeline``

You should be able to see each user’s timeline stored under the directory of ``timeline/{userID}.json``.
