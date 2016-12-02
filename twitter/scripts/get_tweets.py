__author__ = 'yi-linghwong'

import time
import tweepy
import json
import os
import sys

class Extractor():

    def __init__(self):

        print ()
        print ("Welcome")

    def connetToApi(self):

        ###################
        # get api keys
        ###################

        apikey = 'api_key' # get your own api keys!
        apisecret = 'api_secret'

        AccessToken	= 'access_token'
        AccessTokenSecret = 'access_token_secret'

        auth = tweepy.OAuthHandler(apikey, apisecret)
        auth.set_access_token(AccessToken, AccessTokenSecret)

        print ("Connecting to twitter API...")

        api = tweepy.API(auth, wait_on_rate_limit=True)

        return api


    def create_user_list(self):

        lines = open(path_to_user_list,'r').readlines()

        users = []

        for line in lines:
            spline = line.replace('\n','').split(',')
            users.append(spline[0])

        print ("Length of user list is "+str(len(users)))

        return users


    def gettweets_by_user(self):

        user_list = self.create_user_list()
        retries = 5
        sleep_time = 50

        for user in user_list:

            print ("Getting tweets for "+str(user))

            user_tweets = []

            for r in range(retries):

                try:

                    rate_limit = api.rate_limit_status()

                    remaining = rate_limit['resources']['statuses']['/statuses/user_timeline']['remaining']
                    reset_time = rate_limit['resources']['statuses']['/statuses/user_timeline']['reset']

                    print (remaining)

                    ##################
                    # get tweets with twitter user_timeline, excluding RTs and Replies
                    ##################

                    tweets=tweepy.Cursor(api.user_timeline,id=user, include_rts=False, exclude_replies=True).items(3200)

                    for t in tweets:

                        #dumps serialises strings into JSON (which is very similar to python's dict)
                        json_str= json.dumps(t._json)

                        #loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
                        data=json.loads(json_str)

                        #################
                        # check if media exists, and which type
                        #################

                        if 'extended_entities' in data:

                            if 'media' in data['extended_entities']:

                                if data['extended_entities']['media'] != []:

                                    length = len(data['extended_entities']['media'])

                                    for n in range(length):

                                        type = data['extended_entities']['media'][n]['type']


                        elif 'entities' in data:

                            if 'urls' in data['entities']:

                                if (data['entities']['urls'] != []):

                                    length = len(data['entities']['urls'])

                                    for n in range(length):

                                        if (data['entities']['urls'][n]['display_url'].startswith('youtu')):
                                            type = 'video'
                                            break

                                        elif (data['entities']['urls'][n]['display_url'].startswith('vine')):
                                            type = 'video'
                                            break

                                        elif (data['entities']['urls'][n]['display_url'].startswith('amp.twimg')):
                                            type = 'video'
                                            break

                                        elif (data['entities']['urls'][n]['display_url'].startswith('snpy.tv')):
                                            type = 'video'
                                            break

                                        elif (data['entities']['urls'][n]['display_url'].startswith('vimeo')):
                                            type = 'video'
                                            break

                                        else:
                                            type = 'no_media'

                                else:
                                    type = 'no_media'

                            else:
                                type = 'no_media'

                        else:
                            type = 'no_media'


                        ################
                        # append list of parameters to tweet list
                        ################

                        user_tweets.append([user,data['created_at'],data['id_str'],str(data['user']['followers_count']),str(data['user']['friends_count']),str(data['retweet_count']),str(data['favorite_count']),'has_'+type,data['text'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])


                    #################
                    # write (append) data to file for each user
                    #################

                    f = open(path_to_store_raw_tweets,'a')

                    # if file is empty create heading
                    if user == 'NASA':
                        f.write('user,created_time,tweet_id,followers,following,retweet,favourite,has_media,message' + '\n')

                        for ut in user_tweets:
                            f.write(','.join(ut) + '\n')

                    else:
                        for ut in user_tweets:
                            f.write(','.join(ut) + '\n')

                    break

                except Exception as e:
                    print('Failed: ' + str(e))
                    time.sleep(sleep_time)



    def gettweets_by_hashtag(self):

        hashtag_list = ['climate', 'climatechange']
        retries = 5
        sleep_time = 50

        for hashtag in hashtag_list:

            print ("Getting tweets for #"+str(hashtag))

            hashtag_tweets = []

            for r in range(retries):

                try:

                    rate_limit = api.rate_limit_status()

                    remaining = rate_limit['resources']['statuses']['/statuses/user_timeline']['remaining']
                    reset_time = rate_limit['resources']['statuses']['/statuses/user_timeline']['reset']

                    print (remaining)

                    #################
                    # get tweets with Twitter search api, EXCLUDING retweets
                    #################

                    tweets=tweepy.Cursor(api.search, q='#'+hashtag+'-filter:retweets', count=100, lang="en").items(1500)
                    #tweets=tweepy.Cursor(api.search, q=hashtag+'-filter:retweets', count=100, lang="en").items(1500)


                    for t in tweets:

                        #dumps serialises strings into JSON (which is very similar to python's dict)
                        json_str= json.dumps(t._json)

                        #loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
                        data=json.loads(json_str)

                        #################
                        # check if media exists, and which type
                        #################

                        if 'extended_entities' in data:

                            if 'media' in data['extended_entities']:

                                if data['extended_entities']['media'] != []:

                                    length = len(data['extended_entities']['media'])

                                    for n in range(length):

                                        type = data['extended_entities']['media'][n]['type']


                        elif 'entities' in data:

                            if 'urls' in data['entities']:

                                if (data['entities']['urls'] != []):

                                    length = len(data['entities']['urls'])

                                    for n in range(length):

                                        if (data['entities']['urls'][n]['display_url'].startswith('youtu')):
                                            type = 'video'
                                            break

                                        elif (data['entities']['urls'][n]['display_url'].startswith('vine')):
                                            type = 'video'
                                            break

                                        elif (data['entities']['urls'][n]['display_url'].startswith('amp.twimg')):
                                            type = 'video'
                                            break

                                        elif (data['entities']['urls'][n]['display_url'].startswith('snpy.tv')):
                                            type = 'video'
                                            break

                                        elif (data['entities']['urls'][n]['display_url'].startswith('vimeo')):
                                            type = 'video'
                                            break

                                        else:
                                            type = 'no_media'

                                else:
                                    type = 'no_media'

                            else:
                                type = 'no_media'

                        else:
                            type = 'no_media'


                        ################
                        # append list of parameters to tweet list
                        ################

                        hashtag_tweets.append([data['user']['screen_name'],data['created_at'],data['id_str'],str(data['user']['followers_count']),str(data['user']['friends_count']),str(data['retweet_count']),str(data['favorite_count']),'has_'+str(type),data['text'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])


                    #################
                    # write (append) data to file for each user
                    #################

                    f = open(path_to_store_raw_tweets_hashtag+str(hashtag)+'.csv','w')

                    header = ['user','created_time','tweet_id','user_follcount','user_following','retweet','favourite','has_media','message']

                    hashtag_tweets.insert(0,header)

                    for ht in hashtag_tweets:
                            f.write(','.join(ht) + '\n')

                    f.close()

                    break

                except Exception as e:
                    print('Failed: ' + str(e))
                    time.sleep(sleep_time)


    def gettweets_by_id(self):


        #id_list = self.get_tweet_id_list()
        retries = 5
        sleep_time = 50

        lines = open(path_to_tweet_id_list,'r').readlines()

        id_list = []

        for line in lines:
            spline = line.replace('\n','')
            id_list.append(spline)

        id_list = ['792892595240185856']

        print ("Length of id list is "+str(len(id_list)))

        print ("Getting tweets for id list...")

        for id in id_list:

            tweets = []

            for r in range(retries):

                try:


                    rate_limit = api.rate_limit_status()

                    remaining = rate_limit['resources']['statuses']['/statuses/user_timeline']['remaining']
                    reset_time = rate_limit['resources']['statuses']['/statuses/user_timeline']['reset']

                    print (remaining)

                    tweet = api.get_status(id=id)

                    #dumps serialises strings into JSON (which is very similar to python's dict)
                    json_str= json.dumps(tweet._json)

                    #loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
                    data=json.loads(json_str)

                    print (data)


                    #################
                    # check if media exists, and which type
                    #################

                    if 'extended_entities' in data:

                        if 'media' in data['extended_entities']:

                            if data['extended_entities']['media'] != []:

                                length = len(data['extended_entities']['media'])

                                for n in range(length):

                                    type = data['extended_entities']['media'][n]['type']


                    elif 'entities' in data:

                        if 'urls' in data['entities']:

                            if (data['entities']['urls'] != []):

                                length = len(data['entities']['urls'])

                                for n in range(length):

                                    if (data['entities']['urls'][n]['display_url'].startswith('youtu')):
                                        type = 'video'
                                        break

                                    elif (data['entities']['urls'][n]['display_url'].startswith('vine')):
                                        type = 'video'
                                        break

                                    elif (data['entities']['urls'][n]['display_url'].startswith('amp.twimg')):
                                        type = 'video'
                                        break

                                    elif (data['entities']['urls'][n]['display_url'].startswith('snpy.tv')):
                                        type = 'video'
                                        break

                                    elif (data['entities']['urls'][n]['display_url'].startswith('vimeo')):
                                        type = 'video'
                                        break

                                    else:
                                        type = 'no_media'

                            else:
                                type = 'no_media'

                        else:
                            type = 'no_media'

                    else:
                        type = 'no_media'


                    ################
                    # append list of parameters to tweet list
                    ################

                    tweets.append([data['user']['screen_name'],data['created_at'],data['id_str'],str(data['user']['followers_count']),str(data['user']['friends_count']),str(data['retweet_count']),str(data['favorite_count']),'has_'+str(type),data['text'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])

                    #################
                    # write (append) data to file for each user
                    #################

                    f = open(path_to_store_tweets_by_id,'a')

                    for t in tweets:
                        f.write(','.join(t) + '\n')

                    f.close()

                    break

                except tweepy.error.TweepError as e:
                    print ('Tweepy error' + str(e))
                    pass

                except Exception as e:
                    print('Failed: ' + str(e))
                    time.sleep(sleep_time)


    def gettweets_by_replies(self):

        ori_tweet_id_list = []

        retries = 5
        sleep_time = 50

        for id in ori_tweet_id_list:

            replies = []

            for r in range(retries):

                try:

                    rate_limit = api.rate_limit_status()

                    remaining = rate_limit['resources']['statuses']['/statuses/user_timeline']['remaining']
                    reset_time = rate_limit['resources']['statuses']['/statuses/user_timeline']['reset']

                    print(remaining)

                    #################
                    # get tweets with Twitter search api, EXCLUDING retweets
                    #################

                    tweets = tweepy.Cursor(api.search, q='@string' , count=100,
                                           lang="en").items(1500)
                    # tweets=tweepy.Cursor(api.search, q=hashtag+'-filter:retweets', count=100, lang="en").items(1500)


                    for t in tweets:

                        # dumps serialises strings into JSON (which is very similar to python's dict)
                        json_str = json.dumps(t._json)

                        # loads deserialises a string and create a python dict, i.e. it parses the JSON to create a python dict
                        data = json.loads(json_str)


                        if 'in_reply_to_status_id_str' in data:

                            if data['in_reply_to_status_id_str'] == id:
                                print (data)


                                #################
                                # check if media exists, and which type
                                #################

                                if 'extended_entities' in data:

                                    if 'media' in data['extended_entities']:

                                        if data['extended_entities']['media'] != []:

                                            length = len(data['extended_entities']['media'])

                                            for n in range(length):
                                                type = data['extended_entities']['media'][n]['type']


                                elif 'entities' in data:

                                    if 'urls' in data['entities']:

                                        if (data['entities']['urls'] != []):

                                            length = len(data['entities']['urls'])

                                            for n in range(length):

                                                if (data['entities']['urls'][n]['display_url'].startswith('youtu')):
                                                    type = 'video'
                                                    break

                                                elif (data['entities']['urls'][n]['display_url'].startswith('vine')):
                                                    type = 'video'
                                                    break

                                                elif (data['entities']['urls'][n]['display_url'].startswith('amp.twimg')):
                                                    type = 'video'
                                                    break

                                                elif (data['entities']['urls'][n]['display_url'].startswith('snpy.tv')):
                                                    type = 'video'
                                                    break

                                                elif (data['entities']['urls'][n]['display_url'].startswith('vimeo')):
                                                    type = 'video'
                                                    break

                                                else:
                                                    type = 'no_media'

                                        else:
                                            type = 'no_media'

                                    else:
                                        type = 'no_media'

                                else:
                                    type = 'no_media'

                                ################
                                # append list of parameters to tweet list
                                ################

                                replies.append([data['user']['screen_name'], data['created_at'], data['in_reply_to_status_id_str'], data['id_str'],
                                                       str(data['user']['followers_count']), str(data['user']['friends_count']),
                                                       str(data['retweet_count']), str(data['favorite_count']),
                                                       'has_' + str(type),
                                                       data['text'].replace('\n', ' ').replace('\r', '').replace('\t',
                                                                                                                 ' ').replace(
                                                           ',', ' ')])

                    #################
                    # write (append) data to file for each user
                    #################

                    f = open(path_to_store_raw_replies, 'w')

                    header = ['user', 'created_time', 'in_reply_to', 'tweet_id', 'user_follcount', 'user_following', 'retweet',
                              'favourite', 'has_media', 'message']

                    replies.insert(0, header)

                    for rp in replies:
                        f.write(','.join(rp) + '\n')

                    f.close()

                    break

                except Exception as e:
                    print('Failed: ' + str(e))
                    time.sleep(sleep_time)


###############
# variables
###############

path_to_user_list = 'path_to_user_list'
path_to_tweet_id_list = 'path_to_tweet_id_list'

path_to_store_raw_tweets = 'path_to_store_raw_tweets'
path_to_store_raw_tweets_hashtag = 'path_to_store_raw_tweets_hashtag'
path_to_store_tweets_by_id = 'path_to_store_tweets_by_id'
path_to_store_raw_replies = 'path_to_store_raw_replies'


if __name__ == '__main__':

    ################
    # connect to Twitter api
    ################

    ext = Extractor()
    api = ext.connetToApi()

    #################
    # get tweets by user
    #################

    #ext.gettweets_by_user()

    #################
    # get tweets by hashtag
    #################

    #ext.gettweets_by_hashtag()

    #################
    # get tweets by id
    #################

    ext.gettweets_by_id()

    #################
    # get tweets by replies
    #################

    #ext.gettweets_by_replies()

