import streamlit as st
import praw
import google.generativeai as genai
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import re
import json
import os
import logging
from dotenv import load_dotenv
import time
from functools import wraps
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
from collections import defaultdict

# Load environment variables
load_dotenv()

# ===== CONFIGURATION - ADJUST THESE SETTINGS =====
GEMINI_MODEL = 'gemini-2.5-flash'  # For summaries only
RATE_LIMIT_DELAY = 5  # Seconds between API calls
NETWORK_DISCOVERY_LIMIT = 50  # Max discovered subreddits
# =================================================

MAX_RETRIES = 3
RETRY_DELAY = 60

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, delay=RATE_LIMIT_DELAY):
        self.delay = delay
        self.last_call_time = 0
    
    def wait(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.delay:
            time.sleep(self.delay - time_since_last_call)
        self.last_call_time = time.time()

rate_limiter = RateLimiter()
api_call_counter = {"count": 0}

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Suppress warnings
logging.getLogger('grpc._cython.cygrpc').setLevel(logging.ERROR)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(max_retries=MAX_RETRIES, base_delay=RETRY_DELAY):
    """Decorator to retry on rate limit errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    rate_limiter.wait()
                    result = func(*args, **kwargs)
                    api_call_counter["count"] += 1
                    return result
                except Exception as e:
                    error_str = str(e)
                    if '429' in error_str or 'quota' in error_str.lower() or 'rate limit' in error_str.lower():
                        retries += 1
                        if retries >= max_retries:
                            logger.error(f"Max retries exceeded")
                            raise
                        wait_time = base_delay * (2 ** (retries - 1))
                        logger.warning(f"Rate limit hit. Retry {retries}/{max_retries} in {wait_time:.1f}s")
                        time.sleep(wait_time)
                    else:
                        raise
            return None
        return wrapper
    return decorator

# Page config
st.set_page_config(page_title="Reddit Insights", page_icon="🔎", layout="wide")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'overall_analysis' not in st.session_state:
    st.session_state.overall_analysis = None
if 'individual_comments' not in st.session_state:
    st.session_state.individual_comments = None
if 'network_data' not in st.session_state:
    st.session_state.network_data = None

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.subheader("📱 Reddit Settings")
subreddit_input = st.sidebar.text_input("Subreddit(s)", value="technology")
keywords = st.sidebar.text_input("Brand/Keywords to Monitor")
min_upvotes = st.sidebar.number_input("Minimum Upvotes", min_value=0, value=50)
num_posts = st.sidebar.slider("Number of Posts", 5, 100, 20)
num_comments = st.sidebar.slider("Comments per Post", 3, 30, 10)
time_filter = st.sidebar.selectbox("Time Period", ["day", "week", "month", "year", "all"])

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Settings")
summary_length = st.sidebar.radio("Summary Length", ["Brief", "Detailed", "Comprehensive"])
enable_sentiment = st.sidebar.checkbox("Enable Sentiment Analysis (VADER)", value=True)
enable_network = st.sidebar.checkbox("Enable Network Discovery", value=True)

st.sidebar.info(f"Sentiment: VADER (instant)\nSummaries: {GEMINI_MODEL}\nDelay: {RATE_LIMIT_DELAY}s")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Visualization Settings")
wordcloud_max_words = st.sidebar.slider("Word Cloud Max Words", 50, 200, 100)
chart_theme = st.sidebar.selectbox("Chart Theme", ["plotly", "plotly_dark"])

# Helper Functions
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER - instant, no API calls"""
    if not text or len(text.strip()) == 0:
        return {"label": "Neutral", "score": 0.5}
    
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    # Map compound score to label and 0-1 scale
    if compound >= 0.05:
        label = "Positive"
        score = 0.5 + (compound * 0.5)  # Maps 0.05-1.0 to 0.525-1.0
    elif compound <= -0.05:
        label = "Negative"
        score = 0.5 + (compound * 0.5)  # Maps -1.0 to -0.05 to 0.0-0.475
    else:
        label = "Neutral"
        score = 0.5
    
    return {
        "label": label,
        "score": round(score, 2),
        "compound": round(compound, 3),
        "raw_scores": scores
    }

def discover_related_subreddits(reddit, keywords_list, limit=NETWORK_DISCOVERY_LIMIT):
    """Discover subreddits where keywords are discussed"""
    discovered = defaultdict(lambda: {"mentions": 0, "posts": [], "comments": 0})
    
    try:
        for keyword in keywords_list[:3]:  # Limit keyword search
            search_results = reddit.subreddit("all").search(keyword, limit=limit, time_filter="month")
            
            for post in search_results:
                sub_name = post.subreddit.display_name
                discovered[sub_name]["mentions"] += 1
                discovered[sub_name]["posts"].append({
                    "title": post.title,
                    "score": post.score,
                    "url": f"https://reddit.com{post.permalink}"
                })
                
                if len(discovered) >= limit:
                    break
            
            if len(discovered) >= limit:
                break
            
            time.sleep(0.5)
    except Exception as e:
        logger.error(f"Discovery error: {e}")
    
    return dict(discovered)

def calculate_network_metrics(G):
    """Calculate network centrality metrics"""
    metrics = {}
    
    try:
        # Degree Centrality
        metrics['degree'] = nx.degree_centrality(G)
        
        # Betweenness Centrality
        metrics['betweenness'] = nx.betweenness_centrality(G)
        
        # Eigenvector Centrality (if graph is connected enough)
        try:
            metrics['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            metrics['eigenvector'] = {node: 0 for node in G.nodes()}
        
        # PageRank
        metrics['pagerank'] = nx.pagerank(G)
        
        # Closeness Centrality (for connected components)
        if nx.is_connected(G.to_undirected()):
            metrics['closeness'] = nx.closeness_centrality(G)
        else:
            metrics['closeness'] = {node: 0 for node in G.nodes()}
        
    except Exception as e:
        logger.error(f"Metrics calculation error: {e}")
    
    return metrics

def build_subreddit_network(posts_data, individual_comments, discovered_subreddits, input_subreddits):
    """Build network graph of subreddit relationships"""
    G = nx.DiGraph()
    edge_weights = defaultdict(int)
    
    # Add input subreddits as main nodes
    for sub in input_subreddits:
        G.add_node(sub, type='input', mentions=0)
    
    # Add discovered subreddits
    for sub, data in discovered_subreddits.items():
        if sub not in input_subreddits:
            G.add_node(sub, type='discovered', mentions=data['mentions'])
    
    # Count mentions in input subreddits
    for post_data in posts_data:
        sub = post_data['post'].subreddit.display_name
        if sub in G.nodes():
            G.nodes[sub]['mentions'] = G.nodes[sub].get('mentions', 0) + 1
    
    for comment in individual_comments:
        sub = comment['subreddit']
        if sub in G.nodes():
            G.nodes[sub]['mentions'] = G.nodes[sub].get('mentions', 0) + 1
    
    # Create edges between input and discovered subreddits
    for input_sub in input_subreddits:
        for disc_sub in discovered_subreddits:
            if disc_sub != input_sub:
                # Edge weight based on keyword co-occurrence
                weight = discovered_subreddits[disc_sub]['mentions']
                if weight > 0:
                    G.add_edge(input_sub, disc_sub, weight=weight)
    
    # Calculate metrics
    metrics = calculate_network_metrics(G)
    
    return G, metrics

def calculate_brand_health_score(post_results, individual_comments_with_sentiment):
    """Calculate brand health metrics"""
    if not post_results and not individual_comments_with_sentiment:
        return None
    
    all_sentiments = []
    for r in post_results:
        if r.get('sentiment'):
            all_sentiments.append(r['sentiment'])
    for c in individual_comments_with_sentiment:
        if c.get('sentiment'):
            all_sentiments.append(c['sentiment'])
    
    if not all_sentiments:
        return None
    
    total_mentions = len(post_results) + len(individual_comments_with_sentiment)
    total_engagement = sum(r['score'] + r['num_comments'] for r in post_results)
    total_engagement += sum(c['comment_score'] for c in individual_comments_with_sentiment)
    
    avg_sentiment = sum(s['score'] for s in all_sentiments) / len(all_sentiments)
    sentiment_labels = [s['label'] for s in all_sentiments]
    positive_ratio = sentiment_labels.count('Positive') / len(sentiment_labels)
    negative_ratio = sentiment_labels.count('Negative') / len(sentiment_labels)
    
    health_score = (positive_ratio * 40) + (min(total_engagement / 1000, 1) * 30) + ((1 - negative_ratio) * 30)
    health_score = min(round(health_score, 1), 100)
    
    return {
        "health_score": health_score,
        "total_mentions": total_mentions,
        "post_mentions": len(post_results),
        "comment_mentions": len(individual_comments_with_sentiment),
        "total_engagement": total_engagement,
        "avg_sentiment": round(avg_sentiment, 2),
        "positive_ratio": round(positive_ratio * 100, 1),
        "negative_ratio": round(negative_ratio * 100, 1),
        "neutral_ratio": round((1 - positive_ratio - negative_ratio) * 100, 1),
        "avg_upvotes": round(sum(r['score'] for r in post_results) / len(post_results), 1) if post_results else 0,
        "avg_comments": round(sum(r['num_comments'] for r in post_results) / len(post_results), 1) if post_results else 0
    }

def fetch_reddit_data_optimized(reddit, subreddit_name, limit, time_filter, min_upvotes, keywords_list, num_comments_per_post):
    """Fetch Reddit data"""
    posts_data = []
    individual_comments = []
    subreddit = reddit.subreddit(subreddit_name)
    
    if keywords_list:
        search_query = " OR ".join(keywords_list)
        posts_generator = subreddit.search(search_query, time_filter=time_filter, limit=limit * 3)
    else:
        posts_generator = subreddit.top(time_filter=time_filter, limit=limit * 2)
    
    for post in posts_generator:
        if post.score < min_upvotes or (post.author and str(post.author).lower() == 'automoderator'):
            continue
        
        post_text = f"{post.title} {post.selftext}".lower()
        post_has_keyword = not keywords_list or any(kw.lower() in post_text for kw in keywords_list)
        
        post.comments.replace_more(limit=0)
        all_comments = list(post.comments)
        
        matching_comments = []
        for comment in all_comments:
            if hasattr(comment, 'body') and comment.author and str(comment.author).lower() != 'automoderator':
                if keywords_list:
                    comment_text = comment.body.lower()
                    for kw in keywords_list:
                        if kw.lower() in comment_text:
                            matching_comments.append({
                                'post_id': post.id,
                                'post_title': post.title,
                                'post_url': f"https://reddit.com{post.permalink}",
                                'comment_body': comment.body,
                                'comment_score': comment.score,
                                'comment_author': str(comment.author),
                                'created_utc': comment.created_utc,
                                'subreddit': subreddit_name,
                                'matched_keyword': kw
                            })
                            break
        
        sorted_comments = sorted(all_comments, key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
        top_comments_data = [
            {"body": c.body, "score": c.score}
            for c in sorted_comments[:num_comments_per_post]
            if hasattr(c, 'body') and c.author and str(c.author).lower() != 'automoderator'
        ]
        
        if post_has_keyword or matching_comments:
            posts_data.append({
                'post': post,
                'top_comments': top_comments_data,
                'all_comments_text': ' '.join([c.body for c in sorted_comments[:num_comments_per_post] if hasattr(c, 'body') and c.author and str(c.author).lower() != 'automoderator']),
                'match_type': 'post' if post_has_keyword else 'comment_only'
            })
            individual_comments.extend(matching_comments)
        
        if len(posts_data) >= limit:
            break
        
        time.sleep(0.1)
    
    return posts_data, individual_comments

@retry_with_exponential_backoff()
def generate_summary_with_retry(prompt, gemini_model):
    """Generate summary"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def analyze_post_with_gemini(post_data, gemini_model, summary_length):
    """Analyze post - summary only (sentiment handled by VADER)"""
    post = post_data['post']
    top_comments = post_data['top_comments']
    all_comments_text = post_data['all_comments_text']
    
    post_content = f"Title: {post.title}\nContent: {post.selftext[:1000]}"
    comments_text = "\n".join([f"- {c['body'][:200]}" for c in top_comments[:5]])
    combined_text = f"{post.title} {post.selftext} {all_comments_text}"
    
    detail_map = {
        "Brief": "in 2-3 sentences",
        "Detailed": "in a comprehensive paragraph",
        "Comprehensive": "with detailed analysis"
    }
    
    summary_prompt = f"""Summarize this Reddit post and comments {detail_map[summary_length]}.

Post: {post_content}
Top Comments: {comments_text}

Summary:"""
    
    try:
        summary = generate_summary_with_retry(summary_prompt, gemini_model)
    except:
        summary = "Summary unavailable"
    
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
        "sentiment": None,
        "top_comments": top_comments,
        "combined_text": combined_text,
        "match_type": post_data['match_type']
    }

@retry_with_exponential_backoff()
def generate_overall_analysis_with_retry(prompt, gemini_model):
    """Generate overall analysis"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def generate_overall_analysis(results, gemini_model, brand_keywords, individual_comments_with_sentiment):
    """Generate comprehensive analysis"""
    if not results and not individual_comments_with_sentiment:
        return None
    
    all_content = []
    for r in results:
        all_content.append(f"POST: {r['title']}\n{r['selftext'][:300]}")
        for c in r['top_comments'][:3]:
            all_content.append(f"COMMENT: {c['body'][:150]}")
    
    for ic in individual_comments_with_sentiment[:30]:
        all_content.append(f"COMMENT: {ic['comment_body'][:200]}")
    
    combined_content = "\n\n".join(all_content[:50])
    brand_health = calculate_brand_health_score(results, individual_comments_with_sentiment)
    
    if not brand_health:
        return None
    
    analysis_prompt = f"""Analyze Reddit discussions about "{brand_keywords}".

METRICS:
Total: {brand_health['total_mentions']} mentions
Sentiment: {brand_health['positive_ratio']}% positive, {brand_health['negative_ratio']}% negative
Health Score: {brand_health['health_score']}/100

DISCUSSIONS:
{combined_content[:4000]}

Provide analysis with:
## Executive Summary
## Key Trends
## Action Items
## Predictions"""

    try:
        analysis_text = generate_overall_analysis_with_retry(analysis_prompt, gemini_model)
    except:
        analysis_text = "Analysis unavailable due to API limits"
    
    return {"analysis": analysis_text, "brand_health": brand_health}

# Main App
st.title("🔎 Reddit Insights")
st.markdown("**AI-Powered Monitoring & Sentiment Analysis with Network Discovery**")

# Check credentials
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "RedditInsightBot/1.0")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not all([reddit_client_id, reddit_client_secret, gemini_api_key]):
    st.error("⚠️ Missing API credentials!")
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
                "overall_analysis": st.session_state.overall_analysis,
                "network_data": st.session_state.network_data
            }, indent=2, default=str),
            file_name=f"reddit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
with col3:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.analysis_results = None
        st.session_state.overall_analysis = None
        st.session_state.individual_comments = None
        st.session_state.network_data = None
        st.rerun()

st.markdown("---")

# Estimate API calls (reduced because VADER is instant)
if keywords or subreddit_input:
    estimated_post_summaries = num_posts
    estimated_total = estimated_post_summaries + 1  # Only summaries + overall analysis
    estimated_time_minutes = (estimated_total * RATE_LIMIT_DELAY) / 60
    
    st.info(f"""
📊 **Estimated:** ~{estimated_total} API calls (summaries only), ~{estimated_time_minutes:.1f} minutes
💡 **Sentiment Analysis:** VADER (instant, no API calls)
🌐 **Network Discovery:** {'Enabled' if enable_network else 'Disabled'}

API calls tracked: {api_call_counter['count']}
    """)

# Run Analysis
if run_analysis:
    api_call_counter["count"] = 0
    
    try:
        with st.spinner("Initializing..."):
            reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
            genai.configure(api_key=gemini_api_key)
            gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        subreddits = [s.strip() for s in subreddit_input.split(',') if s.strip()]
        keywords_list = [k.strip() for k in keywords.split(',') if k.strip()] if keywords else []
        
        all_results = []
        all_individual_comments = []
        all_posts_data = []
        
        # Fetch data from INPUT subreddits only
        for subreddit_name in subreddits:
            st.info(f"Searching r/{subreddit_name}...")
            
            posts_data, individual_comments = fetch_reddit_data_optimized(
                reddit, subreddit_name, num_posts, time_filter, min_upvotes, keywords_list, num_comments
            )
            
            all_posts_data.extend(posts_data)
            
            if posts_data:
                progress_bar = st.progress(0)
                for idx, post_data in enumerate(posts_data):
                    result = analyze_post_with_gemini(post_data, gemini_model, summary_length)
                    all_results.append(result)
                    progress_bar.progress((idx + 1) / len(posts_data))
                progress_bar.empty()
            
            all_individual_comments.extend(individual_comments)
        
        # VADER Sentiment Analysis (instant!)
        if enable_sentiment and all_results:
            with st.spinner("Analyzing sentiment with VADER..."):
                for result in all_results:
                    sentiment = analyze_sentiment_vader(result['combined_text'])
                    result['sentiment'] = sentiment
        
        if all_individual_comments and enable_sentiment:
            with st.spinner("Analyzing comment sentiment..."):
                for comment in all_individual_comments:
                    sentiment = analyze_sentiment_vader(comment['comment_body'])
                    comment['sentiment'] = sentiment
        
        # Network Discovery (discovers OTHER subreddits)
        discovered_subreddits = {}
        if enable_network and keywords_list:
            with st.spinner("Discovering related subreddits..."):
                discovered_subreddits = discover_related_subreddits(reddit, keywords_list)
                st.info(f"Discovered {len(discovered_subreddits)} related subreddits")
        
        # Build network
        network_data = None
        if discovered_subreddits or all_posts_data:
            with st.spinner("Building network graph..."):
                G, metrics = build_subreddit_network(
                    all_posts_data, all_individual_comments, discovered_subreddits, subreddits
                )
                network_data = {
                    "graph": nx.node_link_data(G),
                    "metrics": {metric_name: {str(k): v for k, v in metric_dict.items()} 
                               for metric_name, metric_dict in metrics.items()},
                    "discovered_subreddits": discovered_subreddits
                }
                st.session_state.network_data = network_data
        
        # Overall analysis
        if all_results or all_individual_comments:
            with st.spinner("Generating analysis..."):
                overall_analysis = generate_overall_analysis(
                    all_results, gemini_model, keywords if keywords else "the topic",
                    all_individual_comments
                )
                st.session_state.overall_analysis = overall_analysis
        
        st.session_state.analysis_results = all_results
        st.session_state.individual_comments = all_individual_comments
        
        st.success(f"✅ Complete! API calls: {api_call_counter['count']} | Sentiment: VADER (instant)")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.exception("Analysis error")

# Display Results
if st.session_state.analysis_results or st.session_state.individual_comments:
    results = st.session_state.analysis_results or []
    individual_comments = st.session_state.individual_comments or []
    overall_analysis = st.session_state.overall_analysis
    network_data = st.session_state.network_data
    
    # Create tabs with Network Analysis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "🤖 AI Insights", 
        "📌 Posts",
        "💬 Comments",
        "📈 Visualizations",
        "📉 Analytics",
        "🌐 Network"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header("📊 Overview")
        
        if overall_analysis and overall_analysis.get('brand_health'):
            brand_health = overall_analysis['brand_health']
            
            col1, col2, col3, col4, col5 = st.columns(5)
            health_color = "🟢" if brand_health['health_score'] >= 70 else "🟡" if brand_health['health_score'] >= 40 else "🔴"
            col1.metric("Health Score", f"{health_color} {brand_health['health_score']}/100")
            col2.metric("Total Mentions", brand_health['total_mentions'])
            col3.metric("Posts", brand_health['post_mentions'])
            col4.metric("Comments", brand_health['comment_mentions'])
            col5.metric("Engagement", f"{brand_health['total_engagement']:,}")
            
            st.markdown("---")
            st.subheader("😊 Sentiment Distribution (VADER)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"🟢 {brand_health['positive_ratio']}%")
            col2.metric("Neutral", f"🟡 {brand_health['neutral_ratio']}%")
            col3.metric("Negative", f"🔴 {brand_health['negative_ratio']}%")
            
            st.markdown("---")
            st.subheader("📈 Engagement")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Upvotes", f"👍 {brand_health['avg_upvotes']}")
            col2.metric("Avg Comments", f"💬 {brand_health['avg_comments']}")
            col3.metric("Avg Sentiment", f"⭐ {brand_health['avg_sentiment']:.2f}/1.0")
    
    # TAB 2: AI Insights
    with tab2:
        st.header("🤖 AI Insights")
        if overall_analysis and overall_analysis.get('analysis'):
            st.markdown(overall_analysis['analysis'])
        else:
            st.info("No insights available")
    
    # TAB 3: Posts
    with tab3:
        st.header("📌 Posts")
        st.markdown(f"**{len(results)}** posts analyzed")
        
        if results:
            col1, col2, col3 = st.columns(3)
            with col1:
                sort_by = st.selectbox("Sort by", ["Upvotes", "Comments", "Sentiment", "Date"])
            with col2:
                sentiment_filter = st.selectbox("Filter", ["All", "Positive", "Neutral", "Negative"])
            with col3:
                match_filter = st.selectbox("Match", ["All", "Post", "Comments"])
            
            filtered_results = results.copy()
            
            if sentiment_filter != "All":
                filtered_results = [r for r in filtered_results if r.get('sentiment') and r['sentiment']['label'] == sentiment_filter]
            
            if match_filter == "Post":
                filtered_results = [r for r in filtered_results if r.get('match_type') == 'post']
            elif match_filter == "Comments":
                filtered_results = [r for r in filtered_results if r.get('match_type') == 'comment_only']
            
            if sort_by == "Upvotes":
                filtered_results.sort(key=lambda x: x['score'], reverse=True)
            elif sort_by == "Comments":
                filtered_results.sort(key=lambda x: x['num_comments'], reverse=True)
            elif sort_by == "Sentiment":
                filtered_results.sort(key=lambda x: x['sentiment']['score'] if x.get('sentiment') else 0, reverse=True)
            elif sort_by == "Date":
                filtered_results.sort(key=lambda x: x['created_utc'], reverse=True)
            
            st.markdown(f"Showing **{len(filtered_results)}** posts")
            st.markdown("---")
            
            for idx, result in enumerate(filtered_results):
                match_badge = "📝 Post" if result.get('match_type') == 'post' else "💬 Comment"
                
                with st.expander(f"**{idx + 1}. {result['title']}** | {match_badge}", expanded=(idx == 0)):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("👍 Upvotes", result['score'])
                    col2.metric("💬 Comments", result['num_comments'])
                    col3.metric("🕐 Posted", datetime.fromtimestamp(result['created_utc']).strftime('%Y-%m-%d'))
                    
                    st.markdown(f"**Author:** u/{result['author']} | **Subreddit:** r/{result['subreddit']}")
                    st.markdown(f"🔗 [View on Reddit]({result['url']})")
                    st.markdown("---")
                    st.markdown("**📝 AI Summary:**")
                    st.info(result['ai_summary'])
                    
                    if result.get('sentiment'):
                        sentiment_color = {'Positive': '🟢', 'Neutral': '🟡', 'Negative': '🔴'}
                        compound = result['sentiment'].get('compound', 0)
                        st.markdown(f"**Sentiment (VADER):** {sentiment_color.get(result['sentiment']['label'], '⚪')} {result['sentiment']['label']} (compound: {compound})")
                    
                    if result.get('selftext'):
                        with st.expander("📄 View Post Content"):
                            st.markdown(result['selftext'])
                    
                    if result.get('top_comments'):
                        st.markdown("**💬 Top Comments:**")
                        for i, comment in enumerate(result['top_comments'][:5], 1):
                            st.markdown(f"{i}. \"{comment['body'][:300]}...\" (+{comment['score']})")
    
    # TAB 4: Comments
    with tab4:
        st.header("💬 Comments")
        st.markdown(f"**{len(individual_comments)}** comments")
        
        if individual_comments:
            col1, col2 = st.columns(2)
            with col1:
                comment_sort = st.selectbox("Sort", ["Score", "Date", "Sentiment"])
            with col2:
                comment_sentiment_filter = st.selectbox("Filter", ["All", "Positive", "Neutral", "Negative"], key="cf")
            
            filtered_comments = individual_comments.copy()
            
            if comment_sentiment_filter != "All":
                filtered_comments = [c for c in filtered_comments if c.get('sentiment') and c['sentiment']['label'] == comment_sentiment_filter]
            
            if comment_sort == "Score":
                filtered_comments.sort(key=lambda x: x['comment_score'], reverse=True)
            elif comment_sort == "Date":
                filtered_comments.sort(key=lambda x: x['created_utc'], reverse=True)
            elif comment_sort == "Sentiment":
                filtered_comments.sort(key=lambda x: x['sentiment']['score'] if x.get('sentiment') else 0, reverse=True)
            
            st.markdown(f"Showing **{len(filtered_comments)}** comments")
            st.markdown("---")
            
            for idx, comment in enumerate(filtered_comments[:100], 1):
                with st.expander(f"**#{idx}** by u/{comment['comment_author']} (↑{comment['comment_score']})", expanded=(idx == 1)):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Score", comment['comment_score'])
                    col2.metric("Subreddit", f"r/{comment['subreddit']}")
                    col3.metric("Date", datetime.fromtimestamp(comment['created_utc']).strftime('%Y-%m-%d'))
                    
                    if comment.get('sentiment'):
                        sentiment_color = {'Positive': '🟢', 'Neutral': '🟡', 'Negative': '🔴'}
                        compound = comment['sentiment'].get('compound', 0)
                        st.markdown(f"**Sentiment:** {sentiment_color.get(comment['sentiment']['label'], '⚪')} {comment['sentiment']['label']} (compound: {compound})")
                    
                    st.markdown(f"**Keyword:** `{comment['matched_keyword']}`")
                    st.markdown(f"**From:** [{comment['post_title'][:80]}...]({comment['post_url']})")
                    st.markdown("---")
                    st.markdown(f"> {comment['comment_body']}")
            
            if len(filtered_comments) > 100:
                st.info(f"Showing top 100 of {len(filtered_comments)} comments")
    
    # TAB 5: Visualizations
    with tab5:
        st.header("📈 Visualizations")
        
        if results or individual_comments:
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Word Cloud", "Sentiment", "Engagement", "Timeline"])
            
            with viz_tab1:
                st.subheader("☁️ Word Cloud")
                all_text = " ".join([f"{r['title']} {r['selftext']} {' '.join([c['body'] for c in r['top_comments']])}" for r in results])
                all_text += " " + " ".join([c['comment_body'] for c in individual_comments])
                
                if all_text.strip():
                    wordcloud = WordCloud(
                        width=1200, height=600, max_words=wordcloud_max_words,
                        background_color='white' if chart_theme == 'plotly' else 'black',
                        colormap='viridis', collocations=False
                    ).generate(clean_text(all_text))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Image(z=wordcloud.to_array()))
                    fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False},
                                    margin={'l': 0, 'r': 0, 't': 30, 'b': 0},
                                    template=chart_theme, title="Most Common Words")
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab2:
                st.subheader("😊 Sentiment Analysis (VADER)")
                
                if enable_sentiment and (results or individual_comments):
                    all_sentiments = []
                    for r in results:
                        if r.get('sentiment'):
                            all_sentiments.append({'source': 'Post', 'sentiment': r['sentiment']['label'], 
                                                  'score': r['sentiment']['score'], 'compound': r['sentiment']['compound']})
                    for c in individual_comments:
                        if c.get('sentiment'):
                            all_sentiments.append({'source': 'Comment', 'sentiment': c['sentiment']['label'], 
                                                  'score': c['sentiment']['score'], 'compound': c['sentiment']['compound']})
                    
                    if all_sentiments:
                        df_all_sentiment = pd.DataFrame(all_sentiments)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_counts = df_all_sentiment['sentiment'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Count']
                            fig = px.pie(sentiment_counts, values='Count', names='Sentiment',
                                       title='Overall Sentiment Distribution',
                                       color='Sentiment',
                                       color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
                                       template=chart_theme)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            sentiment_by_source = df_all_sentiment.groupby(['source', 'sentiment']).size().reset_index(name='count')
                            fig = px.bar(sentiment_by_source, x='source', y='count', color='sentiment',
                                       title='Sentiment by Source',
                                       color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
                                       template=chart_theme, barmode='group')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("VADER Compound Score Distribution")
                        fig = px.histogram(df_all_sentiment, x='compound', nbins=30,
                                         title='VADER Compound Score Distribution',
                                         labels={'compound': 'Compound Score (-1 to 1)'},
                                         template=chart_theme, color='source',
                                         color_discrete_map={'Post': 'blue', 'Comment': 'orange'})
                        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                        time_data = []
                        for r in results:
                            if r.get('sentiment'):
                                time_data.append({'Time': datetime.fromtimestamp(r['created_utc']),
                                                'Sentiment Score': r['sentiment']['score'],
                                                'Compound': r['sentiment']['compound'],
                                                'Title': r['title'][:40] + '...', 'Type': 'Post'})
                        for c in individual_comments:
                            if c.get('sentiment'):
                                time_data.append({'Time': datetime.fromtimestamp(c['created_utc']),
                                                'Sentiment Score': c['sentiment']['score'],
                                                'Compound': c['sentiment']['compound'],
                                                'Title': c['comment_body'][:40] + '...', 'Type': 'Comment'})
                        
                        if time_data:
                            df_time = pd.DataFrame(time_data).sort_values('Time')
                            fig = px.scatter(df_time, x='Time', y='Compound', hover_data=['Title'],
                                           title='Sentiment Over Time (VADER Compound Score)', template=chart_theme,
                                           color='Type', color_discrete_map={'Post': 'blue', 'Comment': 'orange'})
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                            fig.add_hline(y=0.05, line_dash="dot", line_color="green", annotation_text="Positive Threshold")
                            fig.add_hline(y=-0.05, line_dash="dot", line_color="red", annotation_text="Negative Threshold")
                            st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab3:
                st.subheader("📊 Engagement")
                
                if results:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        df_engagement = pd.DataFrame([{'Title': r['title'][:40] + '...', 'Upvotes': r['score']}
                                                     for r in sorted(results, key=lambda x: x['score'], reverse=True)[:15]])
                        fig = px.bar(df_engagement, x='Upvotes', y='Title', title='Top 15 by Upvotes',
                                   template=chart_theme, color='Upvotes', color_continuous_scale='Blues', orientation='h')
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        df_comments = pd.DataFrame([{'Title': r['title'][:40] + '...', 'Comments': r['num_comments']}
                                                   for r in sorted(results, key=lambda x: x['num_comments'], reverse=True)[:15]])
                        fig = px.bar(df_comments, x='Comments', y='Title', title='Top 15 by Comments',
                                   template=chart_theme, color='Comments', color_continuous_scale='Greens', orientation='h')
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    df_correlation = pd.DataFrame([{'Upvotes': r['score'], 'Comments': r['num_comments'], 
                                                   'Title': r['title'][:30] + '...'} for r in results])
                    fig = px.scatter(df_correlation, x='Upvotes', y='Comments', hover_data=['Title'],
                                   title='Upvotes vs Comments Correlation', template=chart_theme,
                                   trendline="ols", color='Comments', color_continuous_scale='Plasma')
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab4:
                st.subheader("📅 Timeline")
                
                if results or individual_comments:
                    timeline_data = []
                    for r in results:
                        timeline_data.append({'Time': datetime.fromtimestamp(r['created_utc']), 'Type': 'Post',
                                            'Title': r['title'][:50] + '...', 'Score': r['score'],
                                            'Comments': r['num_comments'], 'Subreddit': r['subreddit']})
                    for c in individual_comments:
                        timeline_data.append({'Time': datetime.fromtimestamp(c['created_utc']), 'Type': 'Comment',
                                            'Title': c['comment_body'][:50] + '...', 'Score': c['comment_score'],
                                            'Comments': 0, 'Subreddit': c['subreddit']})
                    
                    df_timeline = pd.DataFrame(timeline_data).sort_values('Time')
                    min_score = df_timeline['Score'].min()
                    df_timeline['BubbleSize'] = df_timeline['Score'] + abs(min_score) + 1 if min_score < 0 else df_timeline['Score'] + 1
                    
                    fig = px.scatter(df_timeline, x='Time', y='Score', hover_data=['Title', 'Subreddit', 'Comments'],
                                   title='Activity Timeline', template=chart_theme, size='BubbleSize',
                                   color='Type', color_discrete_map={'Post': 'blue', 'Comment': 'orange'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    df_timeline['Date'] = df_timeline['Time'].dt.date
                    daily_activity = df_timeline.groupby(['Date', 'Type']).size().reset_index(name='Count')
                    fig = px.bar(daily_activity, x='Date', y='Count', color='Type',
                               title='Daily Activity Distribution', template=chart_theme,
                               color_discrete_map={'Post': 'blue', 'Comment': 'orange'})
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 6: Analytics
    with tab6:
        st.header("📉 Analytics")
        
        if results or individual_comments:
            analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["Subreddits", "Keywords", "Users"])
            
            with analytics_tab1:
                st.subheader("📱 Subreddit Breakdown (Input Subreddits)")
                subreddit_stats = {}
                
                for r in results:
                    sub = r['subreddit']
                    if sub not in subreddit_stats:
                        subreddit_stats[sub] = {'posts': 0, 'total_score': 0, 'total_comments': 0, 'comments': 0}
                    subreddit_stats[sub]['posts'] += 1
                    subreddit_stats[sub]['total_score'] += r['score']
                    subreddit_stats[sub]['total_comments'] += r['num_comments']
                
                for c in individual_comments:
                    sub = c['subreddit']
                    if sub not in subreddit_stats:
                        subreddit_stats[sub] = {'posts': 0, 'total_score': 0, 'total_comments': 0, 'comments': 0}
                    subreddit_stats[sub]['comments'] += 1
                
                df_subreddit = pd.DataFrame([{
                    'Subreddit': f"r/{sub}", 'Posts': stats['posts'], 'Comments': stats['comments'],
                    'Total': stats['posts'] + stats['comments'],
                    'Avg Upvotes': round(stats['total_score'] / stats['posts'], 1) if stats['posts'] > 0 else 0,
                    'Avg Comments': round(stats['total_comments'] / stats['posts'], 1) if stats['posts'] > 0 else 0
                } for sub, stats in subreddit_stats.items()]).sort_values('Total', ascending=False)
                
                st.dataframe(df_subreddit, use_container_width=True, hide_index=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(df_subreddit, values='Total', names='Subreddit',
                               title='Mentions by Subreddit', template=chart_theme)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(df_subreddit, x='Subreddit', y=['Posts', 'Comments'],
                               title='Posts vs Comments', template=chart_theme, barmode='group')
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with analytics_tab2:
                st.subheader("🔍 Keyword Performance")
                keyword_stats = {}
                
                for c in individual_comments:
                    kw = c['matched_keyword']
                    if kw not in keyword_stats:
                        keyword_stats[kw] = {'mentions': 0, 'total_score': 0, 'sentiments': []}
                    keyword_stats[kw]['mentions'] += 1
                    keyword_stats[kw]['total_score'] += c['comment_score']
                    if c.get('sentiment'):
                        keyword_stats[kw]['sentiments'].append(c['sentiment']['label'])
                
                if keyword_stats:
                    df_keywords = pd.DataFrame([{
                        'Keyword': kw, 'Mentions': stats['mentions'],
                        'Avg Score': round(stats['total_score'] / stats['mentions'], 1),
                        'Positive %': round(stats['sentiments'].count('Positive') / len(stats['sentiments']) * 100, 1) if stats['sentiments'] else 0,
                        'Negative %': round(stats['sentiments'].count('Negative') / len(stats['sentiments']) * 100, 1) if stats['sentiments'] else 0
                    } for kw, stats in keyword_stats.items()]).sort_values('Mentions', ascending=False)
                    
                    st.dataframe(df_keywords, use_container_width=True, hide_index=True)
                    
                    fig = px.bar(df_keywords, x='Keyword', y='Mentions',
                               title='Keyword Mention Frequency', template=chart_theme,
                               color='Mentions', color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No keyword statistics available")
            
            with analytics_tab3:
                st.subheader("👥 Top Contributors")
                user_stats = {}
                
                for r in results:
                    author = r['author']
                    if author not in user_stats:
                        user_stats[author] = {'posts': 0, 'total_score': 0, 'total_comments': 0, 'comments': 0}
                    user_stats[author]['posts'] += 1
                    user_stats[author]['total_score'] += r['score']
                    user_stats[author]['total_comments'] += r['num_comments']
                
                for c in individual_comments:
                    author = c['comment_author']
                    if author not in user_stats:
                        user_stats[author] = {'posts': 0, 'total_score': 0, 'total_comments': 0, 'comments': 0}
                    user_stats[author]['comments'] += 1
                
                df_users = pd.DataFrame([{
                    'User': f"u/{user}", 'Posts': stats['posts'], 'Comments': stats.get('comments', 0),
                    'Total Activity': stats['posts'] + stats.get('comments', 0),
                    'Total Score': stats['total_score'],
                    'Avg Score': round(stats['total_score'] / stats['posts'], 1) if stats['posts'] > 0 else 0
                } for user, stats in user_stats.items()])
                df_users = df_users.sort_values('Total Activity', ascending=False).head(20)
                
                st.markdown("**Top 20 Most Active Users**")
                st.dataframe(df_users, use_container_width=True, hide_index=True)
                
                fig = px.bar(df_users.head(10), x='User', y=['Posts', 'Comments'],
                           title='Top 10 Users by Activity', template=chart_theme, barmode='stack')
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 7: Network Analysis
    with tab7:
        st.header("🌐 Network Analysis")
        
        if network_data:
            st.info("**Network Discovery:** Shows INPUT subreddits and DISCOVERED subreddits where your keywords appear")
            
            G_data = network_data['graph']
            metrics = network_data['metrics']
            discovered = network_data['discovered_subreddits']
            
            # Network overview
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Input Subreddits", len([n for n in G_data['nodes'] if n.get('type') == 'input']))
            col2.metric("Discovered Subreddits", len([n for n in G_data['nodes'] if n.get('type') == 'discovered']))
            col3.metric("Total Nodes", len(G_data['nodes']))
            col4.metric("Connections", len(G_data['links']))
            
            st.markdown("---")
            
            network_tab1, network_tab2, network_tab3 = st.tabs(["Network Graph", "Centrality Metrics", "Discovered Subreddits"])
            
            with network_tab1:
                st.subheader("📊 Subreddit Network Graph")
                
                # Prepare network visualization data
                G_viz = nx.node_link_graph(G_data)
                pos = nx.spring_layout(G_viz, k=2, iterations=50)
                
                edge_trace = []
                for edge in G_viz.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace.append(
                        go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                 mode='lines',
                                 line=dict(width=0.5 + edge[2].get('weight', 1) * 0.1, color='#888'),
                                 hoverinfo='none',
                                 showlegend=False)
                    )
                
                node_trace_input = go.Scatter(
                    x=[], y=[], text=[], mode='markers+text',
                    hoverinfo='text',
                    marker=dict(size=[], color='blue', line_width=2),
                    textposition="top center",
                    name='Input Subreddits'
                )
                
                node_trace_discovered = go.Scatter(
                    x=[], y=[], text=[], mode='markers',
                    hoverinfo='text',
                    marker=dict(size=[], color='orange', line_width=1),
                    name='Discovered Subreddits'
                )
                
                for node in G_viz.nodes(data=True):
                    x, y = pos[node[0]]
                    node_data = node[1]
                    mentions = node_data.get('mentions', 0)
                    size = 10 + mentions * 2
                    
                    if node_data.get('type') == 'input':
                        node_trace_input['x'] += tuple([x])
                        node_trace_input['y'] += tuple([y])
                        node_trace_input['text'] += tuple([f"r/{node[0]}"])
                        node_trace_input['marker']['size'] += tuple([size])
                    else:
                        node_trace_discovered['x'] += tuple([x])
                        node_trace_discovered['y'] += tuple([y])
                        node_trace_discovered['text'] += tuple([f"r/{node[0]}<br>Mentions: {mentions}"])
                        node_trace_discovered['marker']['size'] += tuple([min(size, 30)])
                
                fig = go.Figure(data=edge_trace + [node_trace_input, node_trace_discovered],
                              layout=go.Layout(
                                  title='Subreddit Network: Input (Blue) vs Discovered (Orange)',
                                  showlegend=True,
                                  hovermode='closest',
                                  margin=dict(b=0,l=0,r=0,t=40),
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  template=chart_theme,
                                  height=700
                              ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with network_tab2:
                st.subheader("📈 Centrality Metrics")
                
                st.markdown("""
                **Understanding Centrality:**
                - **Degree Centrality**: Number of connections (popularity)
                - **Betweenness Centrality**: Bridge between communities
                - **PageRank**: Overall importance (Google's algorithm)
                - **Eigenvector Centrality**: Influence based on connections
                """)
                
                if metrics:
                    metric_choice = st.selectbox("Select Metric", 
                                                 ["degree", "betweenness", "pagerank", "eigenvector", "closeness"])
                    
                    if metric_choice in metrics:
                        metric_data = metrics[metric_choice]
                        df_metrics = pd.DataFrame([
                            {'Subreddit': f"r/{sub}", 'Score': round(score, 4), 
                             'Type': 'Input' if any(n['id'] == sub and n.get('type') == 'input' for n in G_data['nodes']) else 'Discovered'}
                            for sub, score in sorted(metric_data.items(), key=lambda x: x[1], reverse=True)[:20]
                        ])
                        
                        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
                        
                        fig = px.bar(df_metrics, x='Subreddit', y='Score', color='Type',
                                   title=f'Top 20 Subreddits by {metric_choice.title()} Centrality',
                                   template=chart_theme,
                                   color_discrete_map={'Input': 'blue', 'Discovered': 'orange'})
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
            
            with network_tab3:
                st.subheader("🔍 Discovered Subreddits Details")
                
                if discovered:
                    df_discovered = pd.DataFrame([
                        {
                            'Subreddit': f"r/{sub}",
                            'Mentions': data['mentions'],
                            'Sample Posts': len(data['posts'][:3])
                        }
                        for sub, data in sorted(discovered.items(), key=lambda x: x[1]['mentions'], reverse=True)[:50]
                    ])
                    
                    st.dataframe(df_discovered, use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    st.subheader("Sample Posts from Discovered Subreddits")
                    
                    selected_sub = st.selectbox("View posts from:", 
                                               [f"r/{sub}" for sub in list(discovered.keys())[:20]])
                    
                    if selected_sub:
                        sub_name = selected_sub.replace('r/', '')
                        if sub_name in discovered:
                            posts = discovered[sub_name]['posts'][:5]
                            for i, post in enumerate(posts, 1):
                                with st.expander(f"{i}. {post['title'][:80]}... (↑{post['score']})"):
                                    st.markdown(f"**Score:** {post['score']}")
                                    st.markdown(f"🔗 [View on Reddit]({post['url']})")
                else:
                    st.info("No discovered subreddits. Enable 'Network Discovery' and add keywords.")
        else:
            st.info("Network analysis not available. Enable 'Network Discovery' in settings and run analysis with keywords.")

else:
    st.info("👈 Configure settings and click **Run Analysis**")
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Configure Reddit Settings**: Enter subreddit(s) and keywords to monitor
    2. **Adjust Filters**: Set minimum upvotes, time period, and number of posts
    3. **Enable AI Analysis**: Choose summary length and enable sentiment analysis
    4. **Run Analysis**: Click the button and wait for results
    5. **Explore Results**: Navigate through organized tabs to view insights
    
    ### 📊 Features
    
    - **AI-Powered Summaries**: Get concise summaries using Gemini
    - **Sentiment Analysis**: VADER for instant, accurate sentiment detection
    - **Brand Health Score**: Track overall brand perception (0-100)
    - **Network Discovery**: Find related subreddits where keywords are discussed
    - **Centrality Metrics**: Analyze subreddit importance and connections
    - **Visual Analytics**: Charts, word clouds, and network graphs
    - **Export Results**: Download complete analysis as JSON
    
    ### 🆕 What's New
    
    - ⚡ **VADER Sentiment**: Instant sentiment analysis (no API calls!)
    - 🌐 **Network Analysis**: Discover related subreddits automatically
    - 📊 **Centrality Metrics**: Degree, Betweenness, PageRank, Eigenvector, Closeness
    - 🔗 **Connection Mapping**: Visualize how subreddits relate to each other
    
    ### 💡 Tips
    
    - **Input Subreddits**: Only analyze data from subreddits YOU specify
    - **Network Discovery**: Automatically finds OTHER subreddits discussing your keywords
    - **Network Graph**: Blue nodes = your input, Orange nodes = discovered communities
    - **VADER Scores**: Compound score ranges from -1 (negative) to +1 (positive)
    """)
