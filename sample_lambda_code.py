#This is a small part of the ETL process for my personal project that I'm
#working on in my free time. 
#The function extracts various features from a video's transcript,
#cleanses it and then stores the extracted features into a nosql database
#(AWS DynamoDB in this case) for analysis.

#The project is a website that i built to help students to prepare for interviews.
#It allows a user to answer a prompted interview question, analyzes the video and
#provides feedback based on the user's words, facial expressions, and voice patterns.
import numpy as np
import boto3
import ast
import json
import re
import string
import re
import string
from math import modf
from datetime import datetime
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize,pos_tag 
#from sklearn.feature_extraction.text import CountVectorizer
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

# Apply a first round of text cleaning techniques
def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets,
     remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text
#overall code
#This entire code block is in one giant function because it's an
#aws lambda function. Might move some parts out of this main function
#as seperate functions and call them instead for better readability.
def lambda_handler(event, context):
    #get the new raw data from s3 when this function is triggered
    input_bucket = event['Records'][0]['s3']['bucket']['name']
    input_key  = event['Records'][0]['s3']['object']['key']
    transcript = s3.get_object(Bucket = input_bucket,
                          Key = input_key)
    transcript = ast.literal_eval(
        transcript['Body'].read().decode('utf-8')
        )
    transcript=transcript['results']
    s3 = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('overall_table')
    #breaks the transcript name into different elements
    parse_key = input_key.split('--')
    video_datetime = parse_key[0]
    video_bucket = parse_key[1]
    video_api_key = parse_key[2]
    video_archive_id = parse_key[3]
    video_id = '--'.join([str(video_api_key),str(video_archive_id)])
    transcript_item = transcript['items']
    word_count_raw = transcript_item
    #Map each word to its timestamp and confidence
    start_times = []
    end_times = []
    words = []
    confidence = []
    for i in range(0,len(word_count_raw)):
        if list(word_count_raw[i].keys())[0]=='start_time':
            result = word_count_raw[i]
            start_times.append(float(result['start_time']))
            end_times.append(float(result['end_time']))
            words.append(result['alternatives'][0]['content'])
            confidence.append(result['alternatives'][0]['confidence'])

    words = [clean_text_round1(word) for x in words]
    words = [clean_text_round2(word) for x in words]
    word_count_df = pd.DataFrame(
                {'start_time':start_times,
                 'end_time':end_times,
                 'word':words,
                 'confidence':confidence})
    for col in word_count_df.drop('word',1).columns:
        word_count_df[col] = round(word_count_df[col].astype('float'),3)
    #print('first_word_count_df')   
    last_word_timestamp = word_count_df.end_time.shift(1)
    word_interval = word_count_df.start_time - last_word_timestamp
    word_count_df['word_interval'] = word_interval
    #Deal with mistakes such as negative time interval. first find them
    negative_interval_position = np.where(word_count_df['word_interval']<0)[0]
    #sometimes the decimals are switched between start-time and end-time
    #or sometimes the two numbers are entirely switched
    if(len(negative_interval_position)>0):
        for j in range(0,len(negative_interval_position)):
            #for each word with negative spoken time, find
            #the start/end time if it and the word before it
            neg_position = negative_interval_position[j]
            prev_position = neg_position-1
            neg_starttime = word_count_df['start_time'
                                         ].iloc[neg_position]
            prev_endtime = word_count_df['end_time'
                                         ].iloc[prev_position]
            neg_endtime = word_count_df['end_time'
                                         ].iloc[neg_position]
            neg_start_decimal = modf(neg_starttime)[0]
            prv_end_decimal = modf(prev_endtime)[0]
            prv_end_int = modf(prev_endtime)[1]
            neg_end_int = modf(neg_endtime)[1]
            #switch them into the right integers
            if(neg_start_decimal>=prv_end_decimal):
                new_starttime = prv_end_int+neg_start_decimal
            elif(neg_start_decimal<prv_end_decimal):
                new_starttime = neg_end_int+neg_start_decimal

            new_interval = new_starttime-prev_endtime
            new_interval = round(new_interval,2)
            if(new_interval<0):
                word_count_df['start_time'].iloc[neg_position] = prev_endtime
                word_count_df['word_interval'].iloc[neg_position] = 0
            else :
                word_count_df['start_time'].iloc[neg_position] = new_starttime
                word_count_df['word_interval'].iloc[neg_position] = new_interval
    word_count_df['word_interval'].fillna(0,inplace = True)

    start_time = word_count_df.start_time.min()
    end_time = word_count_df.end_time.max()
    overall_time = end_time-start_time
    word_per_second = round(word_count_df.shape[0]/overall_time,2)
    #calculate the trend of taking speed per second
    word_count_df['talking_time'] = word_count_df.end_time-start_time
    temp_min = word_count_df.start_time.min()
    word_count_df['start_time_int'] = word_count_df['start_time'].astype('int')
    second_word_counts = word_count_df['start_time'].astype('int').value_counts()
    second_word_counts.name = 'word_per_seconds'
    word_count_df = word_count_df.merge(second_word_counts,
               how = 'left',
               left_on = 'start_time_int',
               right_index = True)
    word_count_df.drop('start_time_int',1,inplace = True)
    print('word_count_df_with_talking_speed')
    word_count_df.reset_index(inplace = True,drop=True)
    print('dealt with neg_starttime')
    
    speaking_time = word_count_df['end_time']-word_count_df['start_time'].values
    word_count_df['speaking_time'] = speaking_time.abs().round(3)
    word_count_df['speaking_time'] = speaking_time
    #record times where the person uses crutch words to stall for time
    crutch_words = ['um','uh','oh','ah']
    #maybe add things such as 'you know','I mean'
    potential_crutch_words = ['and','like','well','basically','so']
    crutch_rows = word_count_df[word_count_df['word'].str.lower().isin(crutch_words)]
    potential_crutch_rows = word_count_df[
        word_count_df['word'].str.lower().isin(potential_crutch_words)]
    crutch_rows = crutch_rows[['start_time',
                               'end_time',
                               'word',
                               'speaking_time']]
    potential_crutch_rows=potential_crutch_rows[['start_time',
                                                 'end_time',
                                                 'word',
                                                 'speaking_time']]
    #Based on the dataset, if a potential crutch word takes more than 0.8 seconds
    #then it's super likely it is actually used as a crutch word
    potential_crutch_rows = potential_crutch_rows[
        potential_crutch_rows['speaking_time']>0.8]
    crutch_rows= pd.concat([crutch_rows,potential_crutch_rows])
    
    total_crutch_time = crutch_rows.speaking_time.sum()
    total_length = word_count_df['talking_time'].max()
    fraction_time_on_crutch = round(total_crutch_time/total_length,3)
    crutch_rows = crutch_rows.astype('str').to_dict('list')
    #record times where the person stops talking for more than 2 seconds
    long_word_intervals = np.where(word_count_df['word_interval']>1)[0]-1
    long_word_intervals = word_count_df.index[long_word_intervals]
    long_word_intervals = word_count_df.loc[long_word_intervals,'start_time'].astype('str').tolist()
    #finally, convert everything into string and store in DynamoDB
    word_count_df = word_count_df.astype('str')
    word_count_df.index = word_count_df.index.astype('str')
    word_count_df_dict = word_count_df.to_dict('list')
    creation_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    update_expression = ['video_bucket',
                     'video_datetime',
                     'transcript_feature_extraction_time',
                     'transcript',
                     'word_count_df',
                     'overall_talking_speed',
                     'crutch_words',
                     'total_crutch_time',
                     'fraction_time_on_crutch',
                     'long_word_intervals'
    ]

    attribute_key = [':val'+str(x) for x in range(0,len(update_expression))]
    attribute_value = [
        video_bucket,video_datetime,
        creation_time,
        transcript,
        word_count_df_dict,
        str(word_per_second),
        crutch_rows,
        str(total_crutch_time),
        str(fraction_time_on_crutch),
        long_word_intervals
    ]
    attribute_dict = dict(zip(attribute_key,attribute_value))
    update_expression = [x+' ='+y for (x,y) in zip(update_expression,attribute_key)]
    update_expression = 'SET '+', '.join(update_expression)
    table.update_item(
        Key={
            'video_id':video_id
        },
        UpdateExpression=update_expression,
        ExpressionAttributeValues=attribute_dict
    )
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
