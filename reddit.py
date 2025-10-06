import streamlit as st
import praw
import google.generativeai as genai
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import re
from collections import Counter
import json
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress gRPC ALTS warning
logging.getLogger('grpc._cython.cygrpc').setLevel(logging.ERROR)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# Page config
st.set_page_config(
    page_title="Reddit Insight Broadcaster",
    page_icon="🔎",
    layout="wide"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'overall_analysis' not in st.session_state:
    st.session_state.overall_analysis = None
if 'individual_comments' not in st.session_state:
    st.session_state.individual_comments = None

# Sidebar Configuration
st.sidebar.title("⚙️ Configuration")

# Reddit Settings
st.sidebar.subheader("📱 Reddit Settings")
subreddit_input = st.sidebar.text_input("Subreddit(s)", value="technology", help="Comma-separated for multiple")
keywords = st.sidebar.text_input("Brand/Keywords to Monitor", help="Comma-separated. E.g., YourBrand, ProductName")
min_upvotes = st.sidebar.number_input("Minimum Upvotes", min_value=0, value=50)
num_posts = st.sidebar.slider("Number of Posts", 5, 100, 20)
num_comments = st.sidebar.slider("Comments per Post", 3, 30, 10)
time_filter = st.sidebar.selectbox("Time Period", ["day", "week", "month", "year", "all"])
search_in_comments = st.sidebar.checkbox("Search in Comments Too", value=True, help="Also search for keywords in comments")
include_comment_only_posts = st.sidebar.checkbox(
    "Include posts where keyword is only in comments",
    value=False,
    help="If unchecked, only posts with keywords in title/body will be shown"
)

st.sidebar.markdown("---")

# AI Settings
st.sidebar.subheader("🤖 AI Settings")
summary_length = st.sidebar.radio("Summary Length", ["Brief", "Detailed", "Comprehensive"])
enable_sentiment = st.sidebar.checkbox("Enable Sentiment Analysis", value=True)

st.sidebar.markdown("---")

# Visualization Settings
st.sidebar.subheader("📊 Visualization Settings")
wordcloud_max_words = st.sidebar.slider("Word Cloud Max Words", 50, 200, 100)
chart_theme = st.sidebar.selectbox("Chart Theme", ["plotly", "plotly_dark"])


# Helper Functions
def clean_text(text):
    """Clean text for analysis"""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()


def analyze_sentiment_gemini(text, gemini_model):
    """Analyze sentiment using Gemini"""
    try:
        prompt = f"""Analyze the overall sentiment of this Reddit discussion (post + comments combined) and respond with ONLY one word: Positive, Neutral, or Negative.

Consider the tone, emotions, and opinions expressed across all the text.

Text: {text[:2000]}

Sentiment:"""
        response = gemini_model.generate_content(prompt)
        sentiment = response.text.strip()
        
        # Map to score
        score_map = {"Positive": 0.8, "Neutral": 0.5, "Negative": 0.2}
        return {
            "label": sentiment if sentiment in score_map else "Neutral",
            "score": score_map.get(sentiment, 0.5)
        }
    except Exception as e:
        return {"label": "Neutral", "score": 0.5}


def calculate_brand_health_score(results, brand_keywords):
    """Calculate brand health metrics - FIXED to cap at 100"""
    if not results:
        return None
    
    total_mentions = len(results)
    total_engagement = sum(r['score'] + r['num_comments'] for r in results)
    avg_sentiment = sum(r['sentiment']['score'] for r in results if r['sentiment']) / len(results) if results else 0
    
    # Calculate sentiment distribution
    sentiments = [r['sentiment']['label'] for r in results if r['sentiment']]
    positive_ratio = sentiments.count('Positive') / len(sentiments) if sentiments else 0
    negative_ratio = sentiments.count('Negative') / len(sentiments) if sentiments else 0
    
    # Brand Health Score (0-100) - FIXED CALCULATION
    health_score = (
        (positive_ratio * 40) +  # 40% weight on positive sentiment
        (min(total_engagement / 1000, 1) * 30) +  # 30% weight on engagement
        ((1 - negative_ratio) * 30)  # 30% weight on low negativity
    )  # Removed * 100 - already a percentage
    
    # Cap at 100 to prevent overflow
    health_score = min(round(health_score, 1), 100)
    
    return {
        "health_score": health_score,
        "total_mentions": total_mentions,
        "total_engagement": total_engagement,
        "avg_sentiment": round(avg_sentiment, 2),
        "positive_ratio": round(positive_ratio * 100, 1),
        "negative_ratio": round(negative_ratio * 100, 1),
        "avg_upvotes": round(sum(r['score'] for r in results) / len(results), 1),
        "avg_comments": round(sum(r['num_comments'] for r in results) / len(results), 1)
    }


def fetch_reddit_posts_and_comments(reddit, subreddit_name, limit, time_filter, min_upvotes, keywords_list, search_in_comments, include_comment_only_posts):
    """Fetch posts from Reddit - FIXED to distinguish post vs comment matches"""
    posts_data = []
    individual_comments = []
    subreddit = reddit.subreddit(subreddit_name)
    
    # Increase the search multiplier to find more posts
    search_multiplier = 10 if limit <= 10 else 5
    
    for post in subreddit.top(time_filter=time_filter, limit=limit * search_multiplier):
        if post.score >= min_upvotes:
            post_has_keyword = False
            comment_has_keyword = False
            matching_comments = []
            
            # Check if keywords are in title or selftext
            if keywords_list:
                post_text = f"{post.title} {post.selftext}".lower()
                if any(kw.lower() in post_text for kw in keywords_list):
                    post_has_keyword = True
                
                # Also search in comments if enabled
                if search_in_comments:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        if hasattr(comment, 'body'):
                            comment_text = comment.body.lower()
                            for kw in keywords_list:
                                if kw.lower() in comment_text:
                                    # Store individual comment mention
                                    individual_comments.append({
                                        'post_title': post.title,
                                        'post_url': f"https://reddit.com{post.permalink}",
                                        'comment_body': comment.body,
                                        'comment_score': comment.score,
                                        'comment_author': str(comment.author),
                                        'created_utc': comment.created_utc,
                                        'subreddit': subreddit_name,
                                        'matched_keyword': kw
                                    })
                                    matching_comments.append(comment)
                                    comment_has_keyword = True
                                    break
            else:
                # If no keywords specified, include all posts
                post_has_keyword = True
            
            # Decide whether to include post based on settings
            should_include = False
            match_type = None
            
            if post_has_keyword:
                should_include = True
                match_type = "post"
            elif comment_has_keyword and include_comment_only_posts:
                should_include = True
                match_type = "comment_only"
            
            if should_include:
                # Store match type in post object for later display
                post.match_type = match_type
                posts_data.append(post)
                
            if len(posts_data) >= limit:
                break
    
    return posts_data, individual_comments


def analyze_post_with_gemini(post, comments, gemini_model, summary_length, enable_sentiment):
    """Analyze a post and its comments using Gemini"""
    # Prepare content - combine post and comments for sentiment
    post_content = f"Title: {post.title}\nContent: {post.selftext[:1000]}"
    comments_text = "\n".join([f"- {c.body[:200]}" for c in comments[:5]])
    
    # Combine all text for sentiment analysis
    combined_text = f"{post.title} {post.selftext} {' '.join([c.body for c in comments])}"
    
    # Determine summary detail
    detail_map = {
        "Brief": "in 2-3 sentences",
        "Detailed": "in a comprehensive paragraph",
        "Comprehensive": "with detailed analysis including key points and implications"
    }
    
    # Generate summary
    summary_prompt = f"""Summarize this Reddit post and its top comments {detail_map[summary_length]}.

Post:
{post_content}

Top Comments:
{comments_text}

Summary:"""
    
    try:
        summary_response = gemini_model.generate_content(summary_prompt)
        summary = summary_response.text.strip()
    except:
        summary = "Unable to generate summary."
    
    # Analyze sentiment of combined post+comments if enabled
    sentiment = None
    if enable_sentiment:
        sentiment = analyze_sentiment_gemini(combined_text[:2000], gemini_model)
    
    # Process comments
    top_comments = []
    for comment in comments:
        top_comments.append({
            "body": comment.body[:200],
            "score": comment.score
        })
    
    # Get match type from post object
    match_type = getattr(post, 'match_type', 'post')
    
    return {
        "id": post.id,
        "title": post.title,
        "url": f"https://reddit.com{post.permalink}",
        "author": str(post.author),
        "subreddit": post.subreddit.display_name,
        "score": post.score,
        "num_comments": post.num_comments,
        "created_utc": post.created_utc,
        "selftext": post.selftext[:500],
        "ai_summary": summary,
        "sentiment": sentiment,
        "top_comments": top_comments,
        "combined_text": combined_text[:1000],
        "match_type": match_type  # Store whether match was in post or only in comments
    }


def generate_overall_analysis(results, gemini_model, brand_keywords, individual_comments):
    """Generate comprehensive analysis of all posts and comments combined"""
    if not results and not individual_comments:
        return None
    
    # Prepare combined content
    all_content = []
    for r in results:
        all_content.append(f"POST: {r['title']}\n{r['selftext'][:300]}")
        for c in r['top_comments']:
            all_content.append(f"COMMENT: {c['body'][:150]}")
    
    # Add individual comment mentions
    for ic in individual_comments[:20]:
        all_content.append(f"COMMENT MENTION: {ic['comment_body'][:200]}")
    
    combined_content = "\n\n".join(all_content[:50])
    
    # Calculate brand metrics
    brand_health = calculate_brand_health_score(results, brand_keywords)
    
    # Generate comprehensive analysis
    analysis_prompt = f"""You are a brand monitoring expert. Analyze this collection of Reddit posts and comments about "{brand_keywords}".

REDDIT DISCUSSIONS:
{combined_content[:4000]}

METRICS:
- Total Post Mentions: {len(results)}
- Total Comment Mentions: {len(individual_comments)}
- Average Sentiment Score: {brand_health['avg_sentiment'] if brand_health else 'N/A'}
- Positive Mentions: {brand_health['positive_ratio'] if brand_health else 'N/A'}%
- Negative Mentions: {brand_health['negative_ratio'] if brand_health else 'N/A'}%
- Average Engagement: {brand_health['avg_upvotes'] if brand_health else 'N/A'} upvotes, {brand_health['avg_comments'] if brand_health else 'N/A'} comments

Provide a comprehensive analysis covering:
1. **Overall Sentiment & Trends**: What's the general mood and trending topics?
2. **Key Themes & Pain Points**: What are people discussing most? Any complaints or issues?
3. **Opportunities**: What positive signals or opportunities exist for the brand?
4. **Threats & Concerns**: What negative signals or risks should be addressed?
5. **Actionable Recommendations**: Specific steps the brand should take based on this feedback
6. **Predictions**: How might sentiment evolve? What to watch for?

Be specific, actionable, and data-driven in your analysis."""

    try:
        response = gemini_model.generate_content(analysis_prompt)
        return {
            "analysis": response.text.strip(),
            "brand_health": brand_health
        }
    except Exception as e:
        return {
            "analysis": f"Unable to generate overall analysis: {str(e)}",
            "brand_health": brand_health
        }


# Main App
st.title("🔎 Reddit Insight Broadcaster")
st.markdown("**AI-Powered Brand Monitoring & Sentiment Analysis for Reddit**")

# Check for environment variables
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "RedditInsightBot/1.0")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not all([reddit_client_id, reddit_client_secret, gemini_api_key]):
    st.error("⚠️ Missing API credentials! Please set up your .env file with:")
    st.code("""REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=RedditInsightBot/1.0
GEMINI_API_KEY=your_gemini_key""")
    st.stop()

# Action buttons
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
with col2:
    if st.session_state.analysis_results or st.session_state.individual_comments:
        download_data = st.download_button(
            "📥 Download Results",
            data=json.dumps({
                "posts": st.session_state.analysis_results,
                "individual_comments": st.session_state.individual_comments,
                "overall_analysis": st.session_state.overall_analysis
            }, indent=2),
            file_name="reddit_brand_analysis.json",
            mime="application/json",
            use_container_width=True
        )
with col3:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.analysis_results = None
        st.session_state.overall_analysis = None
        st.session_state.individual_comments = None
        st.rerun()

st.markdown("---")

# Run Analysis
if run_analysis:
    try:
        # Initialize APIs
        with st.spinner("🔧 Initializing APIs..."):
            reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
            genai.configure(api_key=gemini_api_key)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Process subreddits
        subreddits = [s.strip() for s in subreddit_input.split(',')]
        keywords_list = [k.strip() for k in keywords.split(',')] if keywords else []
        
        all_results = []
        all_individual_comments = []
        
        for subreddit_name in subreddits:
            st.info(f"📱 Searching r/{subreddit_name} for mentions...")
            
            # Fetch posts and individual comments
            posts, individual_comments = fetch_reddit_posts_and_comments(
                reddit, subreddit_name, num_posts, 
                time_filter, min_upvotes, keywords_list, search_in_comments, include_comment_only_posts
            )
            
            all_individual_comments.extend(individual_comments)
            
            if not posts and not individual_comments:
                st.warning(f"No mentions found in r/{subreddit_name} matching criteria")
                continue
            
            st.success(f"Found {len(posts)} posts and {len(individual_comments)} comment mentions in r/{subreddit_name}")
            
            if posts:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, post in enumerate(posts):
                    status_text.text(f"Analyzing post {idx + 1}/{len(posts)}: {post.title[:50]}...")
                    
                    # Fetch comments
                    post.comments.replace_more(limit=0)
                    top_comments = sorted(post.comments, key=lambda x: x.score, reverse=True)[:num_comments]
                    
                    # Analyze with Gemini
                    result = analyze_post_with_gemini(
                        post, top_comments, gemini_model, 
                        summary_length, enable_sentiment
                    )
                    all_results.append(result)
                    
                    progress_bar.progress((idx + 1) / len(posts))
                
                progress_bar.empty()
                status_text.empty()
        
        # Generate overall analysis
        if all_results or all_individual_comments:
            with st.spinner("🧠 Generating comprehensive brand analysis..."):
                overall_analysis = generate_overall_analysis(
                    all_results, gemini_model, keywords if keywords else "the topic",
                    all_individual_comments
                )
                st.session_state.overall_analysis = overall_analysis
        
        st.session_state.analysis_results = all_results
        st.session_state.individual_comments = all_individual_comments
        
        total_mentions = len(all_results) + len(all_individual_comments)
        st.success(f"✅ Analysis complete! Found {total_mentions} total mentions ({len(all_results)} posts, {len(all_individual_comments)} comments)")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Display Results
if st.session_state.analysis_results or st.session_state.individual_comments:
    results = st.session_state.analysis_results or []
    individual_comments = st.session_state.individual_comments or []
    overall_analysis = st.session_state.overall_analysis
    
    # Overall Brand Analysis - Show First!
    if overall_analysis:
        st.header("🎯 Overall Brand Analysis & Recommendations")
        
        brand_health = overall_analysis['brand_health']
        
        if brand_health:
            # Brand Health Score
            col1, col2, col3, col4 = st.columns(4)
            
            health_color = "🟢" if brand_health['health_score'] >= 70 else "🟡" if brand_health['health_score'] >= 40 else "🔴"
            col1.metric("Brand Health Score", f"{health_color} {brand_health['health_score']}/100")
            col2.metric("Post Mentions", len(results))
            col3.metric("Comment Mentions", len(individual_comments))
            col4.metric("Total Engagement", f"{brand_health['total_engagement']:,}")
        
        st.markdown("---")
        
        # AI Analysis
        st.markdown("### 🤖 AI-Generated Insights & Recommendations")
        st.markdown(overall_analysis['analysis'])
        
        st.markdown("---")
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    total_comments = sum(r['num_comments'] for r in results) if results else 0
    avg_sentiment = sum(r['sentiment']['score'] for r in results if r['sentiment']) / len(results) if results else 0
    most_active = max(results, key=lambda x: x['num_comments']) if results else None
    
    col1.metric("Posts Analyzed", len(results))
    col2.metric("Comment Mentions", len(individual_comments))
    col3.metric("Avg Sentiment", f"{avg_sentiment*100:.0f}% Positive" if results else "N/A")
    col4.metric("Most Discussed", f"{most_active['num_comments']} comments" if most_active else "N/A")
    
    st.markdown("---")
    
    # Individual Comment Mentions Section
    if individual_comments:
        st.subheader("💬 Individual Comment Mentions")
        st.markdown(f"Found **{len(individual_comments)}** comments mentioning your keywords")
        
        # Sort by score
        sorted_comments = sorted(individual_comments, key=lambda x: x['comment_score'], reverse=True)
        
        # Display in a table
        df_comments = pd.DataFrame([
            {
                'Comment': c['comment_body'][:150] + '...' if len(c['comment_body']) > 150 else c['comment_body'],
                'Score': c['comment_score'],
                'Author': c['comment_author'],
                'Post': c['post_title'][:50] + '...' if len(c['post_title']) > 50 else c['post_title'],
                'Subreddit': c['subreddit'],
                'Date': datetime.fromtimestamp(c['created_utc']).strftime('%Y-%m-%d'),
                'URL': c['post_url']
            }
            for c in sorted_comments[:50]  # Show top 50
        ])
        
        st.dataframe(
            df_comments[['Comment', 'Score', 'Author', 'Post', 'Subreddit', 'Date']],
            use_container_width=True,
            hide_index=True
        )
        
        # Expandable view for detailed comments
        with st.expander("📋 View Detailed Comment Mentions"):
            for idx, comment in enumerate(sorted_comments[:20], 1):
                st.markdown(f"""
**{idx}. Comment by u/{comment['comment_author']}** (↑{comment['comment_score']})  
*In post: [{comment['post_title'][:80]}...]({comment['post_url']})*  
*Subreddit: r/{comment['subreddit']} | {datetime.fromtimestamp(comment['created_utc']).strftime('%Y-%m-%d %H:%M')}*

> {comment['comment_body']}

---
""")
        
        st.markdown("---")
    
    # Visualizations (only if we have post results)
    if results:
        st.subheader("📈 Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Word Cloud", "Sentiment Analysis", "Engagement Metrics", "Timeline"])
        
        with tab1:
            # Word Cloud
            all_text = " ".join([
                f"{r['title']} {r['selftext']} {' '.join([c['body'] for c in r['top_comments']])}"
                for r in results
            ])
            
            # Add comment mentions text
            all_text += " " + " ".join([c['comment_body'] for c in individual_comments])
            
            if all_text.strip():
                wordcloud = WordCloud(
                    width=800, height=400,
                    max_words=wordcloud_max_words,
                    background_color='white' if chart_theme == 'plotly' else 'black',
                    colormap='viridis'
                ).generate(clean_text(all_text))
                
                fig = go.Figure()
                fig.add_trace(go.Image(z=wordcloud.to_array()))
                fig.update_layout(
                    xaxis={'visible': False},
                    yaxis={'visible': False},
                    margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                    template=chart_theme
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Sentiment Distribution
            if enable_sentiment:
                sentiment_counts = Counter([r['sentiment']['label'] for r in results if r['sentiment']])
                df_sentiment = pd.DataFrame(sentiment_counts.items(), columns=['Sentiment', 'Count'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        df_sentiment, values='Count', names='Sentiment',
                        title='Overall Sentiment Distribution',
                        color='Sentiment',
                        color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
                        template=chart_theme
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Sentiment over time
                    df_sentiment_time = pd.DataFrame([
                        {
                            'Time': datetime.fromtimestamp(r['created_utc']),
                            'Sentiment Score': r['sentiment']['score'],
                            'Title': r['title'][:30] + '...'
                        }
                        for r in results if r['sentiment']
                    ])
                    
                    fig = px.scatter(
                        df_sentiment_time, x='Time', y='Sentiment Score',
                        hover_data=['Title'],
                        title='Sentiment Trend Over Time',
                        template=chart_theme,
                        color='Sentiment Score',
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sentiment analysis is disabled")
        
        with tab3:
            # Engagement Metrics
            df_engagement = pd.DataFrame([
                {'Title': r['title'][:30] + '...', 'Upvotes': r['score']}
                for r in results
            ])
            
            fig = px.bar(
                df_engagement, x='Title', y='Upvotes',
                title='Post Engagement (Upvotes)',
                template=chart_theme
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Comment Volume
            df_comments = pd.DataFrame([
                {'Title': r['title'][:30] + '...', 'Comments': r['num_comments']}
                for r in results
            ])
            
            fig = px.bar(
                df_comments, x='Title', y='Comments',
                title='Discussion Volume (Comments)',
                template=chart_theme,
                color='Comments',
                color_continuous_scale='Blues'
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Timeline
            df_timeline = pd.DataFrame([
                {
                    'Time': datetime.fromtimestamp(r['created_utc']),
                    'Title': r['title'][:30] + '...',
                    'Score': r['score'],
                    'Comments': r['num_comments']
                }
                for r in results
            ])
            df_timeline = df_timeline.sort_values('Time')
            
            fig = px.scatter(
                df_timeline, x='Time', y='Score',
                hover_data=['Title', 'Comments'],
                title='Post Timeline & Engagement',
                template=chart_theme,
                size='Comments',
                color='Score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Individual Post Analysis
        st.subheader("📌 Individual Post Analysis")
        
        for idx, result in enumerate(results):
            # Add match type badge
            match_badge = "📝 Keyword in Post" if result.get('match_type') == 'post' else "💬 Keyword in Comments Only"
            
            with st.expander(f"**{idx + 1}. {result['title']}** | {match_badge}", expanded=(idx == 0)):
                col1, col2, col3 = st.columns(3)
                col1.metric("👍 Upvotes", result['score'])
                col2.metric("💬 Comments", result['num_comments'])
                col3.metric("🕐 Posted", datetime.fromtimestamp(result['created_utc']).strftime('%Y-%m-%d %H:%M'))
                
                st.markdown(f"**Author:** u/{result['author']} | **Subreddit:** r/{result['subreddit']}")
                st.markdown(f"🔗 [View on Reddit]({result['url']})")
                
                st.markdown("---")
                
                st.markdown("**📝 AI Summary:**")
                st.info(result['ai_summary'])
                
                if result['sentiment']:
                    sentiment_color = {
                        'Positive': '🟢',
                        'Neutral': '🟡',
                        'Negative': '🔴'
                    }
                    st.markdown(f"**😊 Overall Discussion Sentiment:** {sentiment_color.get(result['sentiment']['label'], '⚪')} {result['sentiment']['label']} ({result['sentiment']['score']*100:.0f}% positive)")
                
                st.markdown("**💬 Top Comments:**")
                for i, comment in enumerate(result['top_comments'], 1):
                    st.markdown(f"{i}. \"{comment['body']}\" (+{comment['score']} upvotes)")

else:
    st.info("👈 Configure settings in the sidebar and click **Run Analysis** to begin!")
    
   