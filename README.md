# Arcade API Tools

This repository contains Python scripts that leverage the Arcade API for various automation tasks. The scripts demonstrate how to use the Arcade API to interact with Twitter (X), analyze sentiment, generate charts, and automate email responses.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Tweet Analyzer](#tweet-analyzer)
  - [Features](#features)
  - [Usage](#usage)
  - [How It Works](#how-it-works)
- [Auto Response](#auto-response)
  - [Features](#features-1)
  - [How It Works](#how-it-works-1)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)

## Requirements

- Python 3.8+
- Arcade API access
- OpenAI API access
- Google account (for auto_response.py)

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/arcade.git
cd arcade
```

2. Install the required packages:
```bash
pip install arcadepy openai python-dotenv matplotlib
```

3. Create a `.env` file in the root directory with the following variables:
```
ARCADE_API_KEY=your_arcade_api_key
ARCADE_USER_ID=your_arcade_user_id
OPENAI_API_KEY=your_openai_api_key
```

## Tweet Analyzer

The `tweet_analyzer.py` script fetches tweets based on a keyword, analyzes their sentiment using OpenAI's GPT-4o model, and generates visualization charts.

### Features

- **Tweet Fetching**: Uses Arcade's `X.SearchRecentTweetsByKeywords` tool to fetch recent tweets containing specified keywords.
- **Sentiment Analysis**: Analyzes each tweet using OpenAI's GPT-4o model to determine:
  - Sentiment score (1-10)
  - Topic categorization
  - Language detection
  - Relevance to the search query
- **Data Visualization**: Generates a combined chart with:
  - Sentiment over time (line chart)
  - Sentiment by topic (bar chart with line overlay)
  - Sentiment by language (bar chart with line overlay)
- **Chart Export**: Saves the generated chart as a local PNG file.

### Usage

```bash
python tweet_analyzer.py --keywords "keyword" --max-results 50
```

Arguments:
- `--keywords`: The keyword(s) to search for in tweets (default: "arcade.ai")
- `--max-results`: Maximum number of tweets to fetch (default: 100)

### How It Works

1. **Tweet Fetching**:
   - Authorizes and executes the Arcade Twitter search tool
   - Retrieves tweets containing the specified keywords

2. **Sentiment Analysis**:
   - For each tweet, calls the OpenAI API to analyze:
     - Sentiment score (1-10)
     - Topic categorization (Pricing, Product Features, Competition, etc.)
     - Language detection
     - Relevance to the search query (High, Medium, Low)

3. **Data Processing**:
   - Groups tweets by date, topic, and language
   - Calculates average sentiment scores for each group
   - Counts the number of tweets in each category

4. **Chart Generation**:
   - Creates a matplotlib chart code with three components:
     - Time series chart showing sentiment over time
     - Bar chart showing tweet count and sentiment by topic
     - Bar chart showing tweet count and sentiment by language
   - Uses Arcade's `CodeSandbox.CreateStaticMatplotlibChart` tool to render the chart

5. **Image Saving**:
   - Decodes the base64-encoded chart image
   - Saves it as a local PNG file (`sentiment_chart.png`)

## Auto Response

The `auto_response.py` script automates email responses to meeting requests using Google's email API through Arcade.

### Features

- **Email Fetching**: Retrieves recent emails from your Gmail inbox
- **Meeting Request Detection**: Uses OpenAI's GPT-4o model to identify emails that contain meeting requests
- **Automated Responses**: Generates and sends personalized responses to meeting requests
- **Email Management**: Marks processed emails as read

### How It Works

1. **Email Retrieval**:
   - Authorizes and executes the Arcade Google email tool
   - Fetches the most recent 30 emails
   - Filters for unread emails in the inbox

2. **Meeting Request Analysis**:
   - For each unread email, uses OpenAI to determine if it contains a meeting request
   - Extracts relevant details like the sender's name and meeting purpose
   - Assigns a confidence level (high, medium, low) to the analysis

3. **Response Generation**:
   - For emails identified as meeting requests with high or medium confidence:
     - Generates a personalized response using the sender's name and meeting purpose
     - Sends the response using Arcade's email reply tool
     - Marks the email as read by removing the UNREAD label

4. **Summary Reporting**:
   - Provides a summary of actions taken:
     - Total emails checked
     - Number of unread emails found
     - Number of meeting requests identified
     - Number of responses sent
     - Number of emails marked as read

## Environment Variables

Both scripts require the following environment variables to be set in a `.env` file:

```
ARCADE_API_KEY=your_arcade_api_key
ARCADE_USER_ID=your_arcade_user_id
OPENAI_API_KEY=your_openai_api_key
```

## Dependencies

- `arcadepy`: For interacting with the Arcade API
- `openai`: For sentiment analysis and meeting request detection
- `python-dotenv`: For loading environment variables
- `matplotlib`: For chart generation (used in the chart code)
- `datetime`: For handling tweet dates
- `base64`: For decoding chart images
- `json`: For handling API responses
