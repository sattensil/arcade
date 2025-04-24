import os
import json
import argparse
import base64
from datetime import datetime, timezone
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
from arcadepy import Arcade

# Load environment variables from .env file
load_dotenv()

# Get API keys and user ID from environment variables
API_KEY = os.getenv("ARCADE_API_KEY")
USER_ID = os.getenv("ARCADE_USER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
client = Arcade(api_key=API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Define topic enum
class Topic(str, Enum):
    """Enum for tweet topics"""
    Pricing = "Pricing"
    Product_Features = "Product Features"
    Competition = "Competition"
    Ease_of_Use = "Ease of Use"
    Startups = "Startups"
    Other = "Other"

class Language(str, Enum):
    """Enum for tweet languages"""
    English = "English"
    Spanish = "Spanish"
    French = "French"
    German = "German"
    Portuguese = "Portuguese"
    Italian = "Italian"
    Japanese = "Japanese"
    Chinese = "Chinese"
    Arabic = "Arabic"
    Russian = "Russian"
    Hindi = "Hindi"
    Unknown = "Unknown"
    Other = "Other"

def fetch_tweets(keywords="arcade.ai", max_results=100):
    """Fetch tweets using Arcade's Twitter search tool"""
    print("🔍 Fetching tweets...")
    
    try:
        auth_response = client.tools.authorize(
            tool_name="X.SearchRecentTweetsByKeywords@0.1.12",
            user_id=USER_ID,
        )
        
        print("Authorization response:", auth_response)
        print("Searching for tweets with keywords: " + keywords)
        
        result = client.tools.execute(
            tool_name="X.SearchRecentTweetsByKeywords@0.1.12",
            input={
                "keywords": [keywords],
                "phrases": [],
                "max_results": str(max_results)
            },
            user_id=USER_ID,
        )
        
        if result.success and hasattr(result.output, 'value') and result.output.value:
            tweets_data = result.output.value
            
            if 'data' in tweets_data and tweets_data['data']:
                print(f"✅ Successfully retrieved {len(tweets_data['data'])} tweets")
                return tweets_data['data']
            else:
                print("No tweets found in the data")
                return []
        else:
            if hasattr(result.output, 'error') and result.output.error:
                print(f"Error: {result.output.error}")
            else:
                print("No tweets found in the response")
            return []
            
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

def analyze_tweet(tweet_text, keywords):
    """Analyze tweet sentiment using OpenAI API"""
    keywords_list = keywords.lower().split()
    tweet_text_lower = tweet_text.lower()
    contains_keyword = any(keyword in tweet_text_lower for keyword in keywords_list)
    
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant that analyzes tweet sentiment."},
                {"role": "user", "content": f"Analyze the sentiment of this tweet and determine if it's related to AI technology, products, or companies (be very inclusive). The tweet contains the keyword '{keywords}' which is an AI product/company. Format your response as JSON with the following fields: sentiment_score (1-10), topic (one of: Pricing, Product_Features, Competition, Ease_of_Use, Startups, Other), language (one of: English, Spanish, French, German, Portuguese, Italian, Japanese, Chinese, Arabic, Russian, Hindi, Other), and on_topic (true/false): '{tweet_text}'"},
            ],
            response_format={"type": "json_object"}
        )
        
        if not response.choices or not response.choices[0].message.content:
            print(f"No valid response from OpenAI API")
            return {
                'sentiment_score': 5,
                'topic': 'Other',
                'language': 'English',
                'on_topic': False
            }
            
        try:
            result = json.loads(response.choices[0].message.content)
            sentiment_score = int(result.get('sentiment_score', 5))
            topic = result.get('topic', 'Other')
            language = result.get('language', 'Unknown')
            on_topic = bool(result.get('on_topic', False))
            
            if contains_keyword:
                on_topic = True
                if topic == 'Other':
                    topic = 'Product_Features'
            
            if topic not in [t.name for t in Topic]:
                topic = 'Other'
            if language not in [l.name for l in Language]:
                language = 'Unknown'
                
        except Exception as parse_error:
            print(f"Error parsing response: {parse_error}")
            return {
                'sentiment_score': 5,
                'topic': 'Other',
                'language': 'Unknown',
                'on_topic': contains_keyword
            }
        
        return {
            'sentiment_score': sentiment_score,
            'topic': topic,
            'language': language,
            'on_topic': on_topic
        }
    except Exception as e:
        print(f"Error analyzing tweet: {e}")
        return {
            'sentiment_score': 5,
            'topic': 'Other',
            'language': 'Unknown',
            'on_topic': contains_keyword
        }
def generate_charts(analyzed_tweets, chart_filename="sentiment_chart.png"):
    """Generate sentiment charts using Arcade's matplotlib tool"""
    print("📈 Generating sentiment charts...")
    
    # Ensure we have data to chart
    if not analyzed_tweets:
        print("No tweets to generate charts from.")
        return None
    
    # Process the data for charts
    date_groups = {}
    topic_groups = {}
    
    # Print all unique dates for debugging
    all_dates = sorted(set([tweet.get('date', '') for tweet in analyzed_tweets]))
    print(f"\nAll unique dates in tweets: {all_dates}")
    
    for tweet in analyzed_tweets:
        # Group by date
        date = tweet.get('date', datetime.now().strftime('%Y-%m-%d'))
        if date not in date_groups:
            date_groups[date] = []
        date_groups[date].append(tweet.get('sentiment_score', 5))
        
        # Group by topic
        topic = tweet.get('topic', 'Other')
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(tweet.get('sentiment_score', 5))
    
    # Debug information removed
    
    # Calculate average sentiment per date
    daily_sentiment = []
    for date, scores in date_groups.items():
        if scores:  # Ensure we have scores for this date
            daily_sentiment.append({
                'date': date,
                'sentiment_score': sum(scores) / len(scores)
            })
    
    # Sort by date
    daily_sentiment.sort(key=lambda x: x['date'])
    
    # Group by language
    language_groups = {}
    for tweet in analyzed_tweets:
        language = tweet.get('language', 'Unknown')
        if language not in language_groups:
            language_groups[language] = []
        language_groups[language].append(tweet.get('sentiment_score', 5))
    
    # Calculate average sentiment per language for chart data
    language_sentiment = []
    for language, scores in language_groups.items():
        if scores:  # Ensure we have scores for this language
            language_sentiment.append({
                'language': language,
                'sentiment_score': sum(scores) / len(scores),
                'count': len(scores)
            })
    
    # Sort by sentiment score (descending)
    language_sentiment.sort(key=lambda x: x['sentiment_score'], reverse=True)
    
    # Group by topic for chart data
    topic_groups_for_chart = {}
    for tweet in analyzed_tweets:
        topic = tweet.get('topic', 'Other')
        if topic not in topic_groups_for_chart:
            topic_groups_for_chart[topic] = []
        topic_groups_for_chart[topic].append(tweet.get('sentiment_score', 5))
    
    # Calculate average sentiment per topic for chart data
    topic_sentiment = []
    for topic, scores in topic_groups_for_chart.items():
        if scores:  # Ensure we have scores for this topic
            topic_sentiment.append({
                'topic': topic,
                'sentiment_score': sum(scores) / len(scores),
                'count': len(scores),
                'language': 'Unknown'  # This field is not used for topic charts
            })
    
    # Sort by sentiment score (descending)
    topic_sentiment.sort(key=lambda x: x['sentiment_score'], reverse=True)
    
    # Create matplotlib code for combined charts
    chart_code = create_combined_chart_code(daily_sentiment, topic_sentiment, language_sentiment=language_sentiment)
    
    # Execute the chart generation tool
    try:
        print("\nExecuting chart generation tool...")
        result = client.tools.execute(
            tool_name="CodeSandbox.CreateStaticMatplotlibChart@1.0.0",
            input={
                "owner": "ArcadeAI",
                "name": "arcade-ai",
                "starred": "true",
                "code": chart_code
            },
            user_id=USER_ID,
        )
        
        # Check if the result was successful
        print(f"Chart generation success: {result.success if hasattr(result, 'success') else 'Unknown'}")
    except Exception as e:
        print(f"Error executing chart generation tool: {e}")
        return None
    
    print("✅ Charts generated successfully!")
    
    # Check if result contains a base64 image
    result_str = str(result)
    
    # Save the chart as an image file
    try:
        # Extract the base64 data
        if "base64," in result_str:
            # This looks like a data URL format
            base64_data = result_str.split("base64,")[1].split('"')[0]
        else:
            # Try to find a large chunk of base64 data
            import re
            base64_pattern = r'[A-Za-z0-9+/=]{100,}'  # Look for long base64 strings
            match = re.search(base64_pattern, result_str)
            if match:
                base64_data = match.group(0)
            else:
                base64_data = None
                print("Could not find base64 image data in the response.")
        
        if base64_data:
            # Decode and save the image
            image_data = base64.b64decode(base64_data)
            with open(chart_filename, "wb") as f:
                f.write(image_data)
            print(f"\n✅ Chart saved to: {os.path.abspath(chart_filename)}")
            print(f"Open this file to view the sentiment analysis chart.")
    except Exception as e:
        print(f"Error saving chart image: {e}")
        print("Raw result from Arcade API:")
        print(result_str[:500] + "..." if len(result_str) > 500 else result_str)
    
    return result

def create_combined_chart_code(daily_sentiment, topic_sentiment, language_sentiment=None):
    """Create matplotlib code for three charts: time series and two bar charts"""
    
    # Ensure we have data to chart
    if not daily_sentiment or not topic_sentiment:
        print("Insufficient data for chart generation")
        # Create a simple chart showing no data
        return """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available for chart generation', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        """
    
    # Extract data for charts
    dates = [item['date'] for item in daily_sentiment]
    sentiment_scores_time = [item['sentiment_score'] for item in daily_sentiment]
    
    # For topic sentiment, we need to count the number of tweets per topic
    topic_counts = {}
    topic_sentiments = {}
    
    # For language sentiment, we need to count the number of tweets per language
    language_counts = {}
    language_sentiments = {}
    
    # First collect all sentiment scores for each topic
    for item in topic_sentiment:
        # Process topic data
        topic = item['topic']
        score = item['sentiment_score']
        
        # Handle topic data
        if topic not in topic_sentiments:
            topic_sentiments[topic] = []
            topic_counts[topic] = 0
        
        topic_sentiments[topic].append(score)
        topic_counts[topic] += 1
    
    # Process language data if provided
    if language_sentiment:
        for item in language_sentiment:
            language = item['language']
            score = item['sentiment_score']
            count = item['count']
            
            language_sentiments[language] = [score]  # We already have the average
            language_counts[language] = count
    
    # Calculate average sentiment per topic
    topic_avg_sentiments = {}
    for topic, scores in topic_sentiments.items():
        topic_avg_sentiments[topic] = sum(scores) / len(scores)
    
    # Calculate average sentiment per language
    language_avg_sentiments = {}
    for language, scores in language_sentiments.items():
        language_avg_sentiments[language] = sum(scores) / len(scores)
    
    # Sort topics by average sentiment (descending)
    sorted_topics = sorted(topic_avg_sentiments.keys(), key=lambda t: topic_avg_sentiments[t], reverse=True)
    
    # Sort languages by average sentiment (descending)
    sorted_languages = sorted(language_avg_sentiments.keys(), key=lambda l: language_avg_sentiments[l], reverse=True)
    
    # Prepare sorted data for the charts
    topics = [item['topic'] for item in topic_sentiment]
    sentiment_scores_topic = [item['sentiment_score'] for item in topic_sentiment]
    topic_tweet_counts = [item['count'] for item in topic_sentiment]
    
    languages = sorted_languages
    sentiment_scores_language = [language_avg_sentiments[l] for l in languages]
    language_tweet_counts = [language_counts[l] for l in languages]
    
    # Format the chart code with the data
    dates_repr = repr(dates)
    sentiment_time_repr = repr(sentiment_scores_time)
    topics_repr = repr(topics)
    sentiment_topic_repr = repr(sentiment_scores_topic)
    topic_counts_repr = repr(topic_tweet_counts)
    languages_repr = repr(languages)
    sentiment_language_repr = repr(sentiment_scores_language)
    language_counts_repr = repr(language_tweet_counts)
    
    # Create the chart code with the data inserted
    code = f"""
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.dates as mdates
    from datetime import datetime
    
    # Create a figure with three subplots: one full-width time series and two bar charts below
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Time series plot (full width)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Two bar charts side by side
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create title
    fig.suptitle('Tweet Sentiment Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # Data for sentiment over time - convert to Python lists
    dates_str = {dates_repr}
    sentiment_scores_time = {sentiment_time_repr}
    
    # Convert string dates to datetime objects
    dates_obj = [datetime.strptime(d, '%m-%d') for d in dates_str]
    
    # First subplot - Sentiment over time (full width)
    ax1.plot(dates_obj, sentiment_scores_time, marker='o', linestyle='-', color='#FF0000', 
             linewidth=2, markersize=8)
    
    # X-axis formatting for better date display
    if len(dates_obj) == 1:
        # For a single date, create a custom locator to show just that date
        single_date = dates_obj[0]
        from datetime import timedelta
        # Set tight limits around the single date
        buffer = timedelta(hours=12)  # Half day buffer on each side
        ax1.set_xlim([single_date - buffer, single_date + buffer])
        # Use the exact date as the only tick
        ax1.set_xticks([single_date])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        # Add text labels for context regions
        ax1.text(0.5, 0.9, 'Positive', transform=ax1.transAxes, ha='center')
        ax1.text(0.5, 0.5, 'Neutral', transform=ax1.transAxes, ha='center')
        ax1.text(0.5, 0.1, 'Negative', transform=ax1.transAxes, ha='center')
    else:
        # For multiple dates
        if (max(dates_obj) - min(dates_obj)).days < 14:
            # For dates within 2 weeks, use DayLocator
            ax1.xaxis.set_major_locator(mdates.DayLocator())
        elif (max(dates_obj) - min(dates_obj)).days < 60:
            # For dates within 2 months, use WeekLocator
            ax1.xaxis.set_major_locator(mdates.WeekLocator())
        elif (max(dates_obj) - min(dates_obj)).days < 730:
            # For dates within 2 years, use MonthLocator
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
        else:
            # For longer periods, use YearLocator
            ax1.xaxis.set_major_locator(mdates.YearLocator())
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Add trend line if we have more than one data point
    if len(dates_str) > 1:
        z = np.polyfit(range(len(dates_str)), sentiment_scores_time, 1)
        p = np.poly1d(z)
        ax1.plot(dates_obj, p(range(len(dates_str))), linestyle='--', color='#000000', linewidth=1.5)
    
    # Customize the first subplot
    ax1.set_title('Sentiment Over Time', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Average Sentiment Score (1-10)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 11)
    
    # Format x-axis dates and ticks
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add horizontal lines for reference
    ax1.axhline(y=5, color='#888888', linestyle='-', alpha=0.3)
    ax1.axhline(y=7.5, color='#888888', linestyle='-', alpha=0.3)
    ax1.axhline(y=2.5, color='#888888', linestyle='-', alpha=0.3)
    
    # Annotate regions
    if dates_obj:
        ax1.text(dates_obj[0], 9, 'Positive', fontsize=10, color='#000000')
        ax1.text(dates_obj[0], 4, 'Neutral', fontsize=10, color='#000000')
        ax1.text(dates_obj[0], 1.5, 'Negative', fontsize=10, color='#000000')
    
    # Data for sentiment by topic
    topics = {topics_repr}
    sentiment_scores_topic = {sentiment_topic_repr}
    topic_tweet_counts = {topic_counts_repr}
    
    # Second subplot - Sentiment scores by topic with tweet count overlay
    # Create a twin axis for the tweet counts
    ax2b = ax2.twinx()
    
    # Plot the line for sentiment scores on the main axis
    line = ax2.plot(topics, sentiment_scores_topic, marker='o', linestyle='-', color='#FF0000', 
             linewidth=2, markersize=8)
    
    # Plot the bars for tweet counts on the secondary axis
    bars = ax2b.bar(topics, topic_tweet_counts, color='#000000', alpha=0.3, edgecolor='#000000', linewidth=1.5)
    
    # Add sentiment score labels
    for i, score in enumerate(sentiment_scores_topic):
        ax2.text(i, score + 0.2, f'{{score:.1f}}', ha='center', va='bottom', color='#FF0000', fontsize=10)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2b.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{{int(height)}}', ha='center', va='bottom', fontsize=10)
    
    # Customize the second subplot
    ax2.set_title('Tweet Count and Sentiment by Topic', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Topic', fontsize=12)
    ax2.set_ylabel('Average Sentiment Score (1-10)', fontsize=12, color='#FF0000')
    ax2b.set_ylabel('Number of Tweets', fontsize=12, color='#000000')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax2.set_ylim(0, 11)  # Sentiment score range
    ax2b.set_ylim(0, max(topic_tweet_counts) * 1.2 if topic_tweet_counts else 1)  # Add 20% headroom
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Data for sentiment by language
    languages = {languages_repr}
    sentiment_scores_language = {sentiment_language_repr}
    language_tweet_counts = {language_counts_repr}
    
    # Third subplot - Sentiment scores by language with tweet count overlay
    # Create a twin axis for the tweet counts
    ax3b = ax3.twinx()
    
    # Plot the line for sentiment scores on the main axis
    line = ax3.plot(languages, sentiment_scores_language, marker='o', linestyle='-', color='#FF0000', 
             linewidth=2, markersize=8)
    
    # Plot the bars for tweet counts on the secondary axis
    bars = ax3b.bar(languages, language_tweet_counts, color='#000000', alpha=0.3, edgecolor='#000000', linewidth=1.5)
    
    # Add sentiment score labels
    for i, score in enumerate(sentiment_scores_language):
        ax3.text(i, score + 0.2, f'{{score:.1f}}', ha='center', va='bottom', color='#FF0000', fontsize=10)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3b.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{{int(height)}}', ha='center', va='bottom', fontsize=10)
    
    # Customize the third subplot
    ax3.set_title('Tweet Count and Sentiment by Language', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Language', fontsize=12)
    ax3.set_ylabel('Average Sentiment Score (1-10)', fontsize=12, color='#FF0000')
    ax3b.set_ylabel('Number of Tweets', fontsize=12, color='#000000')
    ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax3.set_ylim(0, 11)  # Sentiment score range
    ax3b.set_ylim(0, max(language_tweet_counts) * 1.2 if language_tweet_counts else 1)  # Add 20% headroom
    
    # Rotate x-axis labels for better readability
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    """
    
    return code

def process_tweets(keywords, max_results):
    """Process tweets for sentiment analysis"""
    tweets = fetch_tweets(keywords, max_results)
    if not tweets:
        print("No tweets found.")
        return None
        
    print(f"Processing {len(tweets)} tweets")
    if tweets:
        print(f"First tweet sample: {str(tweets[0])[:150]}...")
        
    analyzed_tweets = []
    for tweet in tweets:
        try:
            if isinstance(tweet, dict):
                tweet_text = tweet.get('text', tweet.get('full_text', tweet.get('content', tweet.get('message', ''))))
            elif isinstance(tweet, str):
                tweet_text = tweet
            
            if not tweet_text:
                print(f"Skipping tweet with no text: {tweet}")
                continue
            
            tweet_id = tweet.get('id') if isinstance(tweet, dict) else ''
            
            print(f"\nAnalyzing tweet: {tweet_text[:100]}..." if len(tweet_text) > 100 else f"\nAnalyzing tweet: {tweet_text}")
            analysis = analyze_tweet(tweet_text, keywords)
            
            if not analysis:
                print("Error: Analysis returned None")
                continue
                
            sentiment_score = analysis.get('sentiment_score', 5)
            topic = analysis.get('topic', 'Other')
            language = analysis.get('language', 'English')
            on_topic = analysis.get('on_topic', False)
            
            relevance = "High" if on_topic and sentiment_score >= 7 else "Medium" if on_topic else "Low"
            
            print(f"Sentiment: {sentiment_score}/10 | Topic: {topic} | Language: {language} | Relevance: {relevance}")
                
            if not on_topic:
                print(f"Skipping tweet: Not relevant to search query")
                continue
            
            tweet_date = extract_date_from_id(tweet_id) if tweet_id else datetime.now()
            
            analyzed_tweet = {
                'id': tweet_id,
                'text': tweet_text,
                'date': tweet_date.strftime('%m-%d'),
                'datetime': tweet_date,
                'sentiment_score': sentiment_score,
                'topic': topic,
                'language': language,
                'on_topic': on_topic,
                'relevance': relevance
            }
            analyzed_tweets.append(analyzed_tweet)
        except Exception as e:
            print(f"Error processing tweet: {e}")
            continue
    
    if not analyzed_tweets:
        print("\n❌ No tweets to analyze after filtering. Try a different keyword.")
        return None
        
    print(f"✅ Successfully analyzed {len(analyzed_tweets)} tweets")
    return analyzed_tweets

def extract_date_from_id(tweet_id):
    """Extract date from Twitter ID using snowflake format"""
    try:
        twitter_epoch = 1288834974657
        timestamp_ms = (int(tweet_id) >> 22) + twitter_epoch
        timestamp_s = timestamp_ms / 1000.0
        from datetime import datetime, timezone
        return datetime.fromtimestamp(timestamp_s)
    except Exception as e:
        print(f"Error extracting date from tweet ID: {e}")
        return datetime.now()

def main():
    """Main function to run the tweet sentiment analyzer"""
    parser = argparse.ArgumentParser(description="Tweet Sentiment Analyzer")
    parser.add_argument(
        "--keywords", 
        type=str, 
        default="arcade.ai", 
        help="Keywords to search for in tweets"
    )
    parser.add_argument(
        "--max-results", 
        type=int, 
        default=100, 
        help="Maximum number of tweets to fetch"
    )
    
    args = parser.parse_args()
    
    print("🔔 Tweet Sentiment Analyzer")
    print("==========================")
    
    print(f"Fetching tweets with keywords: {args.keywords}")
    
    analyzed_tweets = process_tweets(args.keywords, args.max_results)
    if analyzed_tweets:
        print("\n📈 Generating sentiment charts...")
        generate_charts(analyzed_tweets, "sentiment_chart.png")
        print("\n✨ Analysis complete!")
    else:
        print("\n❌ No tweets to analyze. Try a different keyword.")

if __name__ == "__main__":
    main()
