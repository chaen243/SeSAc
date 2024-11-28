import time
import tweepy
import pandas as pd

# API v2 Client 객체 생성
client = tweepy.Client(
    bearer_token='AAAAAAAAAAAAAAAAAAAAAHrQwwEAAAAAii37vtfhBilc7y8S%2F%2BPv0AADPvo%3DcfjmWif5FBGBvxZbgEADLhKBBS58Aa5jUvUSu5JtirejzMXT3O',
    consumer_key='uMUFd03HFQv5Gsy0kjOUclohC',
    consumer_secret='uFJxUPb54jSnNJYeqQLCLXBTcWMrGwaqIaGB1cOGL0FVyhiNr1',
    access_token='1823024512352595969-Z1rVtuMYTJAHQV7JgaPElwV3Y81EBM',
    access_token_secret='et4sFznHzEox1hraZVqsFaA5c0SPJfjCMBrn5mwitVXzI'
)

# 트윗을 수집하는 함수
def fetch_tweets_with_retry(client, query, max_results=100, retries=3, delay=900):
    attempt = 0
    while attempt < retries:
        try:
            tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=['created_at', 'author_id', 'public_metrics'])
            return tweets
        except Exception as e:
            if '429' in str(e):
                print(f"Rate limit exceeded. Waiting for {delay / 60} minutes...")
                time.sleep(delay)
                attempt += 1
            else:
                raise e
    raise Exception("Failed to fetch tweets after multiple retries.")

# '#news' 해시태그가 포함된 트윗 100개 수집
tweets = fetch_tweets_with_retry(client, "#news lang:en", max_results=100)

# 트윗 데이터를 리스트에 저장
tweets_data = []
if tweets.data:
    for tweet in tweets.data:
        created_at = tweet.created_at if hasattr(tweet, 'created_at') else None
        author_id = tweet.author_id if hasattr(tweet, 'author_id') else None
        text = tweet.text if hasattr(tweet, 'text') else ""
        like_count = tweet.public_metrics['like_count'] if 'like_count' in tweet.public_metrics else 0
        retweet_count = tweet.public_metrics['retweet_count'] if 'retweet_count' in tweet.public_metrics else 0

        tweets_data.append([created_at, author_id, text, like_count, retweet_count])

# 데이터프레임으로 변환
df = pd.DataFrame(tweets_data, columns=['시간', '사용자', '트윗 내용', '좋아요 수', '리트윗 수'])

# 데이터 확인
print(df.head())