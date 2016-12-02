__author__ = 'yi-linghwong'

import facebook
import json
import urllib
import os
import sys
import time

class Extractor_fb():

    def create_user_list(self):

        ################
        # Construct user list from txt file
        ################

        lines = open(path_to_user_list, 'r').readlines()

        user_list = []

        for line in lines:
            spline = line.replace('\n', '').split(',')
            user_list.append(spline[0])

        return user_list

    def connectToApi(self, token):

        print("Connecting to Facebook Graph API")

        graph = facebook.GraphAPI(access_token=access_token) # get your own access token!

        return graph

    def create_post_list(self, user, posts, post_list):

        ##############
        # creates a list of post information we want from the dict returned through API
        ##############

        # get number of page likes for that user

        get_page_likes = graph.get_object(id=user, fields='likes')
        page_likes = str(get_page_likes['likes'])

        #################
        # check if 'message' and 'shares' exist as keys, sometimes they don't (e.g. when page publishes a 'story' such as updating their profile pic)
        #################

        temp_list = []

        for m in range(len(posts['data'])):

            message = 'message'
            shares = 'shares'

            if message in posts['data'][m] and shares in posts['data'][m]:
                print(posts['data'][m]['id'])
                post_list.append([user,posts['data'][m]['created_time'],page_likes, posts['data'][m]['id'],str(posts['data'][m]['likes']['summary']['total_count']),str(posts['data'][m]['shares']['count']),
                                  str(posts['data'][m]['comments']['summary']['total_count']),posts['data'][m]['type'],posts['data'][m]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])
                temp_list.append([user,posts['data'][m]['created_time'],page_likes, posts['data'][m]['id'],
                                  str(posts['data'][m]['likes']['summary']['total_count']),str(posts['data'][m]['shares']['count']),str(posts['data'][m]['comments']['summary']['total_count']), posts['data'][m]['type'],
                                  posts['data'][m]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])

            elif message in posts['data'][m] and shares not in posts['data'][m]:
                print("No shares for " + str(posts['data'][m]['id']))
                post_list.append([user,posts['data'][m]['created_time'],page_likes,posts['data'][m]['id'],str(posts['data'][m]['likes']['summary']['total_count']),str(0),str(posts['data'][m]['comments']['summary']['total_count']),posts['data'][m]['type'],
                                  posts['data'][m]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])
                temp_list.append([user,posts['data'][m]['created_time'],page_likes,posts['data'][m]['id'],str(posts['data'][m]['likes']['summary']['total_count']),str(0),str(posts['data'][m]['comments']['summary']['total_count']),posts['data'][m]['type'],
                                  posts['data'][m]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])


        ##################
        # write to file each time this function is called, so that we don't lose data if error occurs
        ##################

        f = open(path_to_store_fb_posts, 'a')

        # if file is empty create heading
        if os.stat(path_to_store_fb_posts).st_size == 0:
            f.write('user,created_time,page_likes,post_id,like_count,share_count,comment_count,type,message' + '\n')
            for tl in temp_list:
                f.write(','.join(tl) + '\n')

        else:
            for tl in temp_list:
                f.write(','.join(tl) + '\n')

        f.close()

        return post_list

    def create_comment_list(self, id, comments, comment_list):

        ##############
        # creates a list of comment information we want from the dict returned through API
        ##############

        temp_list = []

        for n in range(len(comments['data'])):

            if comments['data'][n]['message'] != '':
                # print (comments['data'][n]['id'])
                comment_list.append([id,comments['data'][n]['created_time'],comments['data'][n]['id'],str(comments['data'][n]['like_count']),str(comments['data'][n]['comment_count']),
                                     comments['data'][n]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',',' ')])
                temp_list.append([id,comments['data'][n]['created_time'],comments['data'][n]['id'],str(comments['data'][n]['like_count']),str(comments['data'][n]['comment_count']),
                                  comments['data'][n]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',',' ')])

        ##################
        # write to file each time this function is called, so that we don't lose data if error occurs
        ##################

        f = open(path_to_store_fb_comments, 'a')

        # if file is empty create heading
        if os.stat(path_to_store_fb_comments).st_size == 0:
            f.write('post_id,created_time,comment_id,like_count,comment_count,message' + '\n')
            for tl in temp_list:
                f.write(','.join(tl) + '\n')

        else:
            for tl in temp_list:
                f.write(','.join(tl) + '\n')

        f.close()

        return comment_list

    def create_comment_list_with_replies(self, id, comments, comment_list):

        ##############
        # creates a list of comment information (including replies to comments) we want from the dict returned through API
        ##############

        temp_list = []
        retries = 5
        sleep_time = 30

        for n in range(len(comments['data'])):

            if comments['data'][n]['comment_count'] == 0:

                print("No replies to this comment, id is " + str(comments['data'][n]['id']))

                if comments['data'][n]['message'] != '':
                    comment_list.append([id,comments['data'][n]['created_time'],comments['data'][n]['id'], str(0),str(comments['data'][n]['like_count']),
                                         str(comments['data'][n]['comment_count']),
                                         comments['data'][n]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])
                    temp_list.append([id,comments['data'][n]['created_time'],comments['data'][n]['id'], str(0),str(comments['data'][n]['like_count']),str(comments['data'][n]['comment_count']),
                                      comments['data'][n]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',',' ')])

            elif comments['data'][n]['comment_count'] > 0:

                print("There are replies to this comment, id is " + str(comments['data'][n]['id']))

                comment_list.append([id,comments['data'][n]['created_time'],comments['data'][n]['id'], str(0),str(comments['data'][n]['like_count']),str(comments['data'][n]['comment_count']),
                                     comments['data'][n]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',',' ')])
                temp_list.append([id,comments['data'][n]['created_time'],comments['data'][n]['id'],str(0),str(comments['data'][n]['like_count']),str(comments['data'][n]['comment_count']),
                                  comments['data'][n]['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',',' ')])

                # check if 'comments' exist as a key. Sometimes even when comment count is greater than 0, there are actually no replies!
                if 'comments' in comments['data'][n]:

                    for m in range(len(comments['data'][n]['comments']['data'])):

                        if comments['data'][n]['comments']['data'][m]['message'] != '':
                            comment_list.append([id,comments['data'][n]['comments']['data'][m]['created_time'],comments['data'][n]['comments']['data'][m]['id'],str(1),
                                                 str(comments['data'][n]['comments']['data'][m]['like_count']), str(0),
                                                 comments['data'][n]['comments']['data'][m]['message'].replace('\n',' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])
                            temp_list.append([id,comments['data'][n]['comments']['data'][m]['created_time'],comments['data'][n]['comments']['data'][m]['id'],str(1),
                                              str(comments['data'][n]['comments']['data'][m]['like_count']),str(0),
                                              comments['data'][n]['comments']['data'][m]['message'].replace('\n',' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])

                    if 'next' in comments['data'][n]['comments']['paging']:

                        for r in range(retries):
                            print("Attempt " + str(r) + " for replies")

                            try:

                                url = comments['data'][n]['comments']['paging']['next']
                                next_url = urllib.request.urlopen(url)
                                readable_page = next_url.read()
                                next_page_comment = json.loads(readable_page.decode())

                                print('2nd page replies with length ' + str(len(next_page_comment['data'])))

                                if len(next_page_comment['data']) != 0:

                                    for x in range(len(next_page_comment['data'])):

                                        if next_page_comment['data'][x]['message'] != '':
                                            comment_list.append([id,next_page_comment['data'][x]['created_time'],
                                                                 next_page_comment['data'][x]['id'], str(1),
                                                                 str(next_page_comment['data'][x]['like_count']),str(0),
                                                                 next_page_comment['data'][x]['message'].replace('\n',' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])
                                            temp_list.append([id,next_page_comment['data'][x]['created_time'],next_page_comment['data'][x]['id'],str(1),
                                                              str(next_page_comment['data'][x]['like_count']),str(0),
                                                              next_page_comment['data'][x]['message'].replace('\n',' ').replace('\r','').replace('\t',' ').replace(',', ' ')])

                                    while 'next' in next_page_comment['paging']:

                                        for x in range(retries):
                                            print("Attempt " + str(x) + " for replies paging")

                                            try:

                                                url = next_page_comment['paging']['next']
                                                next_url = urllib.request.urlopen(url)
                                                readable_page = next_url.read()
                                                next_page_comment = json.loads(readable_page.decode())

                                                print('More replies with length ' + str(len(next_page_comment['data'])))

                                                for y in range(len(next_page_comment['data'])):

                                                    if next_page_comment['data'][y]['message'] != '':
                                                        comment_list.append(
                                                            [id,next_page_comment['data'][y]['created_time'],next_page_comment['data'][y]['id'],str(1),str(next_page_comment['data'][y]['like_count']),str(0),
                                                             next_page_comment['data'][y]['message'].replace('\n',' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])
                                                        temp_list.append(
                                                            [id,next_page_comment['data'][y]['created_time'],next_page_comment['data'][y]['id'],str(1),str(next_page_comment['data'][y]['like_count']),str(0),
                                                             next_page_comment['data'][y]['message'].replace('\n',' ').replace('\r', '').replace('\t',' ').replace(',', ' ')])
                                                break

                                            except urllib.error.HTTPError as e:
                                                print("HTTPError caught, retrying...", e.read())
                                                time.sleep(sleep_time)

                                            except Exception as e:
                                                print('Failed: ' + str(e))
                                                time.sleep(sleep_time)

                                break


                            except urllib.error.HTTPError as e:
                                print("HTTPError caught, retrying...", e.read())
                                time.sleep(sleep_time)

                            except Exception as e:
                                print('Failed: ' + str(e))
                                time.sleep(sleep_time)

        f = open(path_to_store_fb_comments_replies, 'a')

        # if file is empty create heading
        if os.stat(path_to_store_fb_comments_replies).st_size == 0:
            f.write('post_id,created_time,comment_id,is_reply,like_count,comment_count,message' + '\n')
            for tl in temp_list:
                f.write(','.join(tl) + '\n')

        else:
            for tl in temp_list:
                f.write(','.join(tl) + '\n')

        f.close()

        return comment_list

    def get_page_posts(self, graph):

        post_limit = 20
        page_limit = 100
        post_list = []
        retries = 5
        sleep_time = 30
        user_list = self.create_user_list()

        for user in user_list:

            try:

                posts = graph.get_connections(id=user, connection_name='posts', limit=post_limit,
                                              fields='shares, message, id, type, created_time, likes.summary(true), comments.summary(true)')  # posts is a dict (with other dicts inside)


                #################
                # 'data' is the dictionary that contains the post messages, which are all in one list: data = {[message 1, message 2, ...]}
                #################

                print("Collecting page 1 for " + user)

                post_list = self.create_post_list(user, posts, post_list)


                ################
                # the next section gets the posts for the first 'next' URL (which is retrieved with the get_connections method), and run only once
                ################


                if 'next' in posts['paging']:

                    for r in range(retries):

                        print("Collecting page 2 for " + user)

                        print("Attempt " + str(r))

                        try:

                            url = posts['paging']['next']
                            next_url = urllib.request.urlopen(url)
                            readable_page = next_url.read()
                            next_page = json.loads(readable_page.decode())

                            post_list = self.create_post_list(user, next_page, post_list)

                            break

                        except urllib.error.HTTPError as e:
                            print("HTTPError caught, retrying...", e.read())
                            time.sleep(sleep_time)

                        except Exception as e:
                            print('Failed: ' + str(e))
                            time.sleep(10)


                            ###############
                            # the next section gets the posts for the second 'next' URL onwards (which are retrieved with the urllib!)
                            ###############

                    for l in range(page_limit - 2):

                        # check if there is a next page
                        if 'paging' in next_page:

                            print("Collecting page " + str(l + 3) + " for " + user)

                            ###############
                            # Try and except to catch HTTPError (Internal server error), set a maximum number of retries
                            ###############

                            for r in range(retries):

                                print("Attempt " + str(r))

                                try:

                                    url = next_page['paging']['next']
                                    next_url = urllib.request.urlopen(url)
                                    readable_page = next_url.read()
                                    next_page = json.loads(readable_page.decode())

                                    post_list = self.create_post_list(user, next_page, post_list)

                                    break

                                except urllib.error.HTTPError as e:
                                    print("HTTPError caught, retrying...", e.read())
                                    time.sleep(sleep_time)

                                except Exception as e:
                                    print('Failed: ' + str(e))
                                    time.sleep(10)

                        else:
                            print("No more next page")
                            break

            except Exception as e:
                print('Failed: ' + str(e))

        return post_list


    def get_comments(self, graph, id_list):

        comment_list = []
        retries = 5
        sleep_time = 50

        for id in id_list:

            ###########
            # get comment count for the post with this id
            ###########

            print("Getting comments for " + str(id))

            try:

                comment_obj = graph.get_object(id=id, fields='comments.summary(true)')
                comment_count = comment_obj['comments']['summary']['total_count']

                print('Comment count is ' + str(comment_count))

                if comment_count <= 100 and comment_count > 0:
                    comments_limit = comment_count
                    # page_limit = 1

                elif comment_count > 100:
                    comments_limit = 100
                    # page_limit = int(comment_count/100)+1

                elif comment_count == 0:
                    print("No comments, skipping")
                    continue

                comments = graph.get_connections(id=id, connection_name='comments', limit=comments_limit,
                                                 fields='message, id, created_time, comments{like_count,message,id,created_time}, like_count, comment_count')
                print(comments)

                ##############
                # 'data' is the dictionary that contains the post messages, which are all in one list: data = {[message 1, message 2, ...]}
                ##############

                print('1st page with length ' + str(len(comments['data'])))

                comment_list = self.create_comment_list(id, comments, comment_list)


                ##############
                # check if there is a next comment page by checking for the key 'next' in the 'comments' dictionary obtained from get_connections method
                ##############

                if 'next' in comments['paging']:

                    for r in range(retries):

                        print("Attempt " + str(r))

                        try:

                            url = comments['paging']['next']
                            next_url = urllib.request.urlopen(url)
                            readable_page = next_url.read()
                            next_page = json.loads(readable_page.decode())

                            print('2nd page with length ' + str(len(next_page['data'])))

                            comment_list = self.create_comment_list(id, next_page, comment_list)

                            ##############
                            # check if there is a next comment page by checking for the key 'next' in the 'next_page' dictionary obtained from URL
                            # Important! While loop has to be under the if statement above
                            ##############

                            while 'next' in next_page['paging']:

                                for x in range(retries):

                                    print("Attempt " + str(x) + " for paging")

                                    try:

                                        url = next_page['paging']['next']
                                        next_url = urllib.request.urlopen(url)
                                        readable_page = next_url.read()
                                        next_page = json.loads(readable_page.decode())

                                        print(len(next_page['data']))

                                        comment_list = self.create_comment_list(id, next_page, comment_list)

                                        break

                                    except urllib.error.HTTPError as e:
                                        print("HTTPError caught, retrying...", e.read())
                                        time.sleep(sleep_time)

                                    except Exception as e:
                                        print('Failed: ' + str(e))
                                        time.sleep(sleep_time)

                            break


                        except urllib.error.HTTPError as e:
                            print("HTTPError caught, retrying...", e.read())
                            time.sleep(sleep_time)

                        except Exception as e:
                            print('Failed: ' + str(e))
                            time.sleep(sleep_time)

            except Exception as e:
                print('Failed: ' + str(e))

        return comment_list

    def get_replies_to_comment(self, graph, id_list):

        comment_list = []
        retries = 5
        sleep_time = 50

        for id in id_list:

            print("###########################Getting comments for " + str(id))

            ###########
            # get comment count for the post with this id
            ###########

            print("Getting comments for " + str(id))

            try:

                comment_obj = graph.get_object(id=id, fields='comments.summary(true)')
                comment_count = comment_obj['comments']['summary']['total_count']

                print('Comment count is ' + str(comment_count))

                if comment_count <= 100 and comment_count > 0:
                    comments_limit = comment_count
                    # page_limit = 1

                elif comment_count > 100:
                    comments_limit = 100
                    # page_limit = int(comment_count/100)+1

                elif comment_count == 0:
                    print("No comments, skipping")
                    continue

                comments = graph.get_connections(id=id, connection_name='comments', limit=comments_limit,
                                                 fields='message, id, created_time, comments{like_count,message,id,created_time}, like_count, comment_count')


                ##############
                # 'data' is the dictionary that contains the post messages, which are all in one list: data = {[message 1, message 2, ...]}
                ##############

                print('1st page with length ' + str(len(comments['data'])))

                comment_list = self.create_comment_list_with_replies(id, comments, comment_list)


                ##############
                # check if there is a next comment page by checking for the key 'next' in the 'comments' dictionary obtained from get_connections method
                ##############

                if 'next' in comments['paging']:

                    for r in range(retries):

                        print("Attempt " + str(r))

                        try:

                            url = comments['paging']['next']
                            next_url = urllib.request.urlopen(url)
                            readable_page = next_url.read()
                            next_page = json.loads(readable_page.decode())

                            print('2nd page with length ' + str(len(next_page['data'])))

                            comment_list = self.create_comment_list_with_replies(id, next_page, comment_list)


                            ##############
                            # check if there is a next comment page by checking for the key 'next' in the 'next_page' dictionary obtained from URL
                            # Important! While loop has to be under the if statement above
                            ##############

                            while 'next' in next_page['paging']:

                                for x in range(retries):

                                    print("Attempt " + str(x) + " for paging")

                                    try:

                                        url = next_page['paging']['next']
                                        next_url = urllib.request.urlopen(url)
                                        readable_page = next_url.read()
                                        next_page = json.loads(readable_page.decode())

                                        print("Another page with length " + str(len(next_page['data'])))

                                        comment_list = self.create_comment_list_with_replies(id, next_page, comment_list)

                                        break

                                    except urllib.error.HTTPError as e:
                                        print("HTTPError caught, retrying...", e.read())
                                        time.sleep(sleep_time)

                                    except Exception as e:
                                        print('Failed: ' + str(e))
                                        time.sleep(sleep_time)

                            break

                        except urllib.error.HTTPError as e:
                            print("HTTPError caught, retrying...", e.read())
                            time.sleep(sleep_time)

                        except Exception as e:
                            print('Failed: ' + str(e))
                            time.sleep(sleep_time)


            except Exception as e:
                print('Failed: ' + str(e))

        print(len(comment_list))

        return comment_list

    def get_post_by_id(self):

        lines = open(path_to_id_list,'r').readlines()

        posts = []

        for line in lines:
            spline = line.rstrip('\n')

            for n in range(5):

                try:

                    post = graph.get_object(id=spline,
                                        fields='from, shares, message, id, type, created_time, likes.summary(true), comments.summary(true)')


                    if 'message' in post:

                        if 'shares' in post:

                            post_by_id = [post['from']['name'],post['created_time'],post['id'],str(post['likes']['summary']['total_count']),
                                      str(post['shares']['count']),
                                      str(post['comments']['summary']['total_count']),post['type'],
                                      post['message'].replace('\n', ' ').replace('\r', '').replace('\t', ' ').replace(',', ' ')]

                        else:

                            post_by_id = [post['from']['name'], post['created_time'], post['id'],
                                          str(post['likes']['summary']['total_count']),
                                          str('0'),
                                          str(post['comments']['summary']['total_count']), post['type'],
                                          post['message'].replace('\n', ' ').replace('\r', '').replace('\t', ' ').replace(',', ' ')]

                    else:
                        print ("no message, skipping")
                        break


                    f = open(path_to_store_post_by_id,'a')

                    if os.stat(path_to_store_post_by_id).st_size == 0:
                        f.write('user,created_time,post_id,like_count,share_count,comment_count,type,message' + '\n')
                        f.write(','.join(post_by_id)+'\n')
                        f.close()

                    else:
                        f.write(','.join(post_by_id)+'\n')
                        f.close()

                    break

                except Exception as e:
                    print('Failed: ' + post['id'] + str(e))
                    time.sleep(5)


            # print([post['created_time'], post['id'], str(post['likes']['summary']['total_count']), str(0),
            #    str(post['comments']['summary']['total_count']), post['type'],
            #    post['message'].replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',',' ')])

    def get_comment_by_id(self, id):

        comment = graph.get_object(id=id, fields='message, id, created_time, comments, comment_count, like_count')

        print(comment)

    def remove_duplicates(self):

        #########
        # remove duplicates by post/comment id
        #########

        target_list = []
        list_clean = []
        temp = []

        lines = open(path_to_store_fb_comments_replies, 'r').readlines()

        # Create a list of lists from a list of strings!
        for line in lines:
            target_list.append(line.strip().replace('\n', ' ').split(', '))

        print("Original length of list is " + str(len(target_list)))

        # check duplicates of comment/post ID
        for tl in target_list:
            if tl[2] not in temp:
                list_clean.append(tl)
                temp.append(tl[2])

        print("New length of list is " + str(len(list_clean)))

        return list_clean

    def create_id_list(self):

        id_list = []

        # lines is a list of strings ['nasa, 2016-01-16, ID, message', 'nasa, 2016-01-01, ID, message', ...]
        lines = open(path_to_store_fb_posts, 'r').readlines()

        for line in lines:
            spline = line.replace("\n", "").split(',')
            # spline = ['nasa', '2016-01-16', 'ID', 'message']

            id_list.append(spline[3])

        del (id_list[0])

        return id_list

###############
# variables
###############

path_to_user_list = 'path_to_user_list'
path_to_store_fb_posts = 'path_to_store_fb_posts'
path_to_store_fb_comments = 'path_to_store_fb_comments'
path_to_store_fb_comments_replies = 'path_to_store_fb_comments_replies'

path_to_id_list = 'path_to_id_list'
path_to_store_post_by_id = 'path_to_store_post_by_id'


if __name__ == '__main__':

    ################
    # connect to Facebook Graph API
    ################

    ext = Extractor_fb()
    graph = ext.connectToApi(access_token)

    ################
    # get posts for pages
    ################

    #posts = ext.get_page_posts(graph)


    ###############
    # get comments for collected posts based on their id's
    ###############

    #ids = ext.create_id_list()
    #comments = ext.get_replies_to_comment(graph, ids)


    ###############
    # remove duplicates
    ###############

    # clean_list = ext.remove_duplicates()


    ###############
    # get post by id
    ###############

    ext.get_post_by_id()

    ###############
    # get single commet
    ###############

    # single_comment = ext.get_comment_by_id('id')
