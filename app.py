import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from scipy.stats import skew

# --------------------------
# Helper Functions
# --------------------------

def load_lottieurl(url: str):
    """
    Load a Lottie animation from a URL.
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def explain_top_movies(top_movies):
    titles = top_movies['title'].tolist()
    counts = top_movies['ratings_count'].tolist()
    explanation = "### Insights on the Most Popular Movies\n"
    explanation += "The bar chart below showcases the top 10 movies that have garnered the highest number of ratings. This popularity often reflects a movie's widespread appeal and cultural impact.\n\n"
    
    for i in range(min(3, len(titles))):
        explanation += f"**{i+1}. {titles[i]}** has received **{counts[i]:,} ratings**. "
        if i == 0:
            explanation += "This top position highlights its immense popularity among viewers, possibly due to compelling storytelling, stellar performances, or significant cultural relevance. "
        elif i == 1:
            explanation += "Securing the second spot indicates strong viewer engagement and positive reception. "
        elif i == 2:
            explanation += "Being in the top three underscores its consistent appeal and likely critical acclaim. "
    
    explanation += "These leading movies set trends and influence viewer preferences, making them central to discussions within the film community."
    return explanation

def explain_ratings_distribution(ratings):
    rating_counts = ratings['rating'].value_counts().sort_index()
    most_common_rating = rating_counts.idxmax()
    most_common_count = rating_counts.max()
    explanation = "### Understanding the Ratings Distribution\n"
    explanation += f"The histogram above reveals that **{most_common_rating} stars** is the most frequently assigned rating, received by **{most_common_count:,} users**. "
    explanation += "This trend suggests that users generally hold movies in high regard, leaning towards favorable evaluations.\n\n"
    explanation += "A skew towards higher ratings can indicate a positive user experience, while a balanced distribution would reflect a wider range of opinions."
    return explanation

def explain_ratings_over_time(ratings_per_year):
    latest_year = ratings_per_year['year'].max()
    latest_year_count = ratings_per_year.loc[ratings_per_year['year'] == latest_year, 'ratings_count'].values[0]
    previous_year = latest_year - 1
    explanation = "### Trends in Movie Ratings Over the Years\n"
    
    if previous_year in ratings_per_year['year'].values:
        previous_year_count = ratings_per_year.loc[ratings_per_year['year'] == previous_year, 'ratings_count'].values[0]
        change = latest_year_count - previous_year_count
        if previous_year_count == 0:
            percent_change = "N/A (no ratings in previous year)"
            change_str = "infinite"
        else:
            percent_change = (change / previous_year_count) * 100
            change_str = f"{percent_change:+.2f}%"
        explanation += f"In **{latest_year}**, there were **{latest_year_count:,} ratings**, which is a **{change_str}** change compared to **{previous_year}** with **{previous_year_count:,} ratings**. "
        
        if percent_change == "N/A (no ratings in previous year)":
            explanation += "This indicates that there were no ratings in the previous year to compare with."
        else:
            if percent_change > 0:
                explanation += "This increase suggests that more users are engaging with rating movies, possibly due to an expanding user base or increased platform activity."
            elif percent_change < 0:
                explanation += "This decrease could point to a decline in user engagement or a shift in user behavior over the years."
            else:
                explanation += "This stable trend shows that user engagement with movie ratings has remained consistent over the years."
    else:
        explanation += f"In **{latest_year}**, there were **{latest_year_count:,} ratings**. There is no data available for the previous year to assess changes in user engagement."
    
    explanation += "\n\nMonitoring these trends helps platforms understand user engagement levels and adjust their strategies accordingly to maintain or boost interaction."
    return explanation

def explain_genre_distribution(genre_counts):
    if genre_counts.shape[0] >= 2:
        top_genre = genre_counts.iloc[0]
        second_genre = genre_counts.iloc[1]
        explanation = "### Distribution of Movie Genres\n"
        explanation += f"The pie chart above illustrates that **{top_genre['genre']}** is the most prevalent genre with **{top_genre['count']:,}** movies, followed by **{second_genre['genre']}** with **{second_genre['count']:,}** movies. "
        explanation += "This dominance indicates viewer preferences and can guide content creation and acquisition strategies to align with popular genres."
    elif genre_counts.shape[0] == 1:
        top_genre = genre_counts.iloc[0]
        explanation = "### Distribution of Movie Genres\n"
        explanation += f"The pie chart above shows that **{top_genre['genre']}** is the sole genre present with **{top_genre['count']:,}** movies. "
        explanation += "A diverse genre distribution is essential for catering to varied viewer tastes."
    else:
        explanation = "### Distribution of Movie Genres\n"
        explanation += "The pie chart does not display any genre distribution as the dataset contains no genres."
    return explanation

def explain_top_rule(rule):
    explanation = (
        f"### Spotlight on the Top Association Rule\n\n"
        f"The top association rule is **{rule['antecedents_str']} → {rule['consequents_str']}** with a **support** of **{rule['support']:.4f}** (**{rule['support'] * 100:.2f}%** of movies), a **confidence** of **{rule['confidence']:.2f}**, and a **lift** of **{rule['lift']:.2f}**.\n\n"
    )
    if rule['lift'] > 1:
        explanation += (
            f"This means that movies in the **{rule['antecedents_str']}** genre are **{rule['lift']:.2f} times** more likely to be associated with the **{rule['consequents_str']}** genre than expected by random chance. "
            f"The high lift value indicates a strong and meaningful relationship between these genres."
        )
    elif rule['lift'] == 1:
        explanation += (
            f"This indicates that there is no association between the **{rule['antecedents_str']}** and **{rule['consequents_str']}** genres beyond what would be expected by random chance."
        )
    else:
        explanation += (
            f"This suggests that the association between **{rule['antecedents_str']}** and **{rule['consequents_str']}** genres is weaker than random chance, indicating a possible inverse relationship."
        )
    return explanation

def explain_scatter_plot(rules):
    average_support = rules['support'].mean()
    average_confidence = rules['confidence'].mean()
    average_lift = rules['lift'].mean()
    explanation = "### Understanding the Support vs. Confidence Scatter Plot\n\n"
    explanation += (
        f"The scatter plot visualizes the relationship between **support** and **confidence** for each association rule. On average, the rules have a **support** of **{average_support:.4f}** (**{average_support * 100:.2f}%** of movies) and a **confidence** of **{average_confidence:.2f}**. "
        f"The size and color of the points represent the **lift** values, where larger and darker points indicate stronger associations.\n\n"
    )
    explanation += (
        f"For instance, a rule with high support and high confidence implies that a large portion of movies contain both genres, and the presence of one genre reliably predicts the other. "
        f"Conversely, rules with low support might represent niche genre pairings, while low confidence suggests that the association is less reliable."
    )
    return explanation

def explain_association_table():
    explanation = "### Dive Deeper with the Interactive Rules Table\n\n"
    explanation += (
        "The table above lists the top 50 association rules sorted by **lift**. Each row represents a rule where the **antecedent** genres (left side) predict the **consequents** genres (right side). "
        "Here's what each metric means:\n\n"
        "- **Support:** How frequently the genre combination appears in the dataset.\n"
        "- **Confidence:** The likelihood that the consequent genre appears when the antecedent genre is present.\n"
        "- **Lift:** The strength of the association compared to random chance.\n\n"
        "By analyzing these metrics, you can identify which genres are commonly enjoyed together, enabling more effective recommendation strategies."
    )
    return explanation

def explain_recommendations(selected_genre, top_associations):
    if not top_associations.empty:
        # Collect associated genres and their lift values
        associations = top_associations[['consequents_str', 'lift']].sort_values(by='lift', ascending=False)
        associations_list = associations.apply(lambda row: f"**{row['consequents_str']}** (lift: {row['lift']:.2f})", axis=1).tolist()
        if len(associations_list) > 1:
            associations_paragraph = ', '.join(associations_list[:-1]) + f", and {associations_list[-1]}."
        else:
            associations_paragraph = associations_list[0]
        
        explanation = "### In-Depth Look at Recommendations\n\n"
        explanation += (
            f"The bar chart above highlights the top genres associated with **{selected_genre}**. Each bar represents an associated genre and its **lift** value, which indicates the strength of the association. "
            f"These associations include {associations_paragraph} "
            f"Users who enjoy **{selected_genre}** movies often find these genres appealing, enhancing their viewing experience with complementary content."
        )
        
        # Consolidate Practical Applications, Solving Challenges, and Empowering Decisions into one paragraph
        explanation += "\n\n"
        explanation += (
            "Addressing the common challenge faced by movie streaming platforms and theaters in recommending content that truly resonates with users, leveraging **Association Rule Mining** can enhance recommendations by identifying genre associations. "
            f"For instance, movie streaming services can refine their algorithms by suggesting complementary genres like {', '.join([assoc.split(' (')[0] for assoc in associations_list])} to users who enjoy **{selected_genre}**, thereby boosting satisfaction and engagement. "
            f"Similarly, movie theaters can curate diverse lineups by pairing genres such as **{selected_genre}** and {', '.join([assoc.split(' (')[0] for assoc in associations_list])}, attracting a wider audience. "
            "Individual users also benefit by discovering new genres that align with their tastes, enriching their viewing experience. "
            "These data-driven strategies not only personalize recommendations but also empower stakeholders to foster greater engagement, loyalty, and platform usage, ensuring that both businesses and users enjoy a more tailored and satisfying movie experience."
        )
        return explanation
    else:
        return "No strong associations found for the selected genre."

# --------------------------
# Streamlit App Layout
# --------------------------

# Set Streamlit page configuration
st.set_page_config(
    page_title="MovieLens 20M Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
        padding-right: 1rem;
        padding-left: 1rem;
    }
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    /* Card Styles */
    .card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        color: #FFFFFF;
        margin-bottom: 20px;
    }
    .card h3 {
        color: #4CAF50;
    }
    /* Slider Styling */
    .custom-slider {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .custom-slider .stSlider > div {
        width: 50%; /* Adjust the width as needed */
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie Animations
welcome_animation_url = "https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"  # Celebration animation
closing_animation_url = "https://assets8.lottiefiles.com/packages/lf20_x62chJ.json"  # Thank you animation

lottie_welcome = load_lottieurl(welcome_animation_url)
lottie_closing = load_lottieurl(closing_animation_url)


# Sidebar Navigation using Horizontal Menu
selected_menu = option_menu(
    menu_title=None,
    options=["Welcome", "Data Overview", "Data Preprocessing", "Exploratory Analysis", "Association Rule Mining", "Recommendations"],
    icons=['house', 'bar-chart-line', 'wrench', 'search', 'pie-chart', 'star'],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "10!important", "background-color": "#262730"},
        "icon": {"color": "white", "font-size": "18px"}, 
        "nav-link": {
            "color": "white", 
            "font-size": "16px", 
            "text-align": "center", 
            "padding": "10",  # Remove padding
            "margin": "0",  # Remove margin
            "display": "flex",
            "align-items": "center",  # Vertically center the tabs
            "justify-content": "center",  # Horizontally center the tabs
            "height": "100%"  # Fill the entire height of the nav bar
        },
        "nav-link-selected": {"background-color": "#4CAF50"},
    }
)
# Load Data Function
@st.cache_data
def load_data():
    """
    Load the complete MovieLens 20M dataset.
    """
    # Adjust the path as necessary
    movies = pd.read_csv('ml-20m/movies.csv')
    ratings = pd.read_csv('ml-20m/ratings.csv')
    tags = pd.read_csv('ml-20m/tags.csv')
    
    return movies, ratings, tags

# Load Data
with st.spinner("Loading data..."):
    movies, ratings, tags = load_data()

# Preprocess Data Function
@st.cache_data
def preprocess_data(movies, ratings, tags):
    # Converting Timestamps to Readable Dates
    ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
    tags['date'] = pd.to_datetime(tags['timestamp'], unit='s')
    
    # Extracting Genres
    movies['genres'] = movies['genres'].str.split('|')
    
    return movies, ratings, tags

movies, ratings, tags = preprocess_data(movies, ratings, tags)

# Explode genres globally
genres_exploded = movies.explode('genres')
genre_counts = genres_exploded['genres'].value_counts().reset_index()
genre_counts.columns = ['genre', 'count']

# One-hot encode genres globally
genres_onehot = genres_exploded.pivot_table(index='movieId', columns='genres', aggfunc='size', fill_value=0)

# Initialize rules variable globally in session state
if "rules" not in st.session_state:
    st.session_state["rules"] = pd.DataFrame()

# --------------------------
# Menu Options
# --------------------------


if selected_menu == "Welcome":
    # Welcome Page with Animation
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 60px;'>Welcome to MovieLens 20M Analysis</h1>
        """,
        unsafe_allow_html=True,
    )

    # Lottie Animation (if available)
    if lottie_welcome:
        st_lottie(lottie_welcome, height=300, key="welcome")
    else:
        st.error("Failed to load the welcome animation.")

    st.markdown(
        """
        **Empowering Movie Recommendations with Data-Driven Insights**  
        Movie streaming platforms and theaters face a common challenge: recommending films that truly resonate with users, thereby increasing engagement and satisfaction. With a vast array of movies available, it can be difficult to consistently suggest content that users will enjoy. This is where **Association Rule Mining** comes in, by analyzing patterns in genre preferences, we can uncover which genres tend to be enjoyed together, providing actionable insights for better recommendations.

        ### Problem at Hand:  
        Movie recommendations are often generic or uninspiring. Users are bombarded with options but struggle to find films that match their taste. Streaming platforms and theaters also face the challenge of presenting engaging content. Our goal is to solve this problem using **data-driven insights** to improve movie recommendations.

        ### Our Solution:  
        By analyzing movie data from **2005 to 2015**, we can tailor movie recommendations that suit different genres, increase user satisfaction, and even expand content diversity. This app allows you to explore recommendations based on genres and movie preferences, helping both users and businesses discover new, exciting movie experiences.

        ### What You Can Explore:
        - **Genre Distributions**: Dive into how different genres are distributed across the dataset.
        - **Ratings Analysis**: Analyze how movies have been rated over the years.
        - **Movie Recommendations**: Explore personalized movie suggestions based on genre associations.

        **Let’s begin this exciting journey of movie discovery!**
        """
    )

    # New section: Meet Our Team
    st.markdown("## Meet Our Team")
    st.markdown(
        """
        We are **Group Nime**, a dedicated team passionate about uncovering insights from the MovieLens 20M dataset. Our collective skills and diverse perspectives drive this interactive data journey. Below are our team members:
        """
    )

    # Team member cards
    team_members = [
        {"name": "Mark Kenneth Badilla", "role": "Leader", "years": "SY 2020-2023", "image": "https://scontent-atl3-2.xx.fbcdn.net/v/t1.15752-9/462542089_1703313630429237_312007961235826364_n.jpg?_nc_cat=105&ccb=1-7&_nc_sid=0024fc&_nc_ohc=RQCRH3lRqdgQ7kNvgEHZ7T7&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent-atl3-2.xx&oh=03_Q7cD1QFZ0fMlDpCtEKrytphI7fJNjyDX9aG7ME-8vR4r334HkA&oe=6777B999"},
        {"name": "Rob Borinaga", "role": "Member", "years": "SY 2019-2023", "image": "https://scontent-atl3-1.xx.fbcdn.net/v/t1.15752-9/462538749_1568925197074088_6661845771934051377_n.jpg?_nc_cat=108&ccb=1-7&_nc_sid=0024fc&_nc_ohc=gWKbZNogNk0Q7kNvgFIyFtT&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent-atl3-1.xx&oh=03_Q7cD1QGcBbJKO-TmOo31A7qEgbMpZ0WAUoiDj4PA8CRnpnLZCw&oe=6777B556"},
        {"name": "Alestair Cyril Coyoca", "role": "Member", "years": "SY 2021-2024", "image": "https://scontent-atl3-1.xx.fbcdn.net/v/t1.15752-9/461838713_1963657944110054_1114849457222079080_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=0024fc&_nc_ohc=KOG-jvTUdpYQ7kNvgGfluOF&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent-atl3-1.xx&oh=03_Q7cD1QFSmOY1sUra6iLPrnlFj7IotJCUnTYo_pfygcvvHu_mMg&oe=6777D927"},
        {"name": "Carmelyn Nime Gerali", "role": "Member", "years": "SY 2018-2022", "image": "https://scontent-atl3-1.xx.fbcdn.net/v/t1.15752-9/462540186_1059839835604866_5455648228103355812_n.jpg?_nc_cat=106&ccb=1-7&_nc_sid=0024fc&_nc_ohc=vqtEpNM3KpMQ7kNvgHlqFTh&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent-atl3-1.xx&oh=03_Q7cD1QH8DcPo29M6b_chOIZwGbcYQ5HDJhYp4TmIxb-tcFcUoQ&oe=6777B777"},
        {"name": "James Alein Ocampo", "role": "Member", "years": "SY 2020-2024", "image": "https://scontent-atl3-1.xx.fbcdn.net/v/t1.15752-9/462537913_496803843176373_4085186211692713416_n.jpg?stp=dst-jpg_s2048x2048&_nc_cat=103&ccb=1-7&_nc_sid=0024fc&_nc_ohc=q5cmf_W0WJsQ7kNvgFnffhl&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent-atl3-1.xx&oh=03_Q7cD1QFdbCijccs5-_V1kQd2cxvObbsPP30jYUcVp1t1MrXwUw&oe=6777A850"}
    ]

    # Create two rows: one for the top 3 cards, another for the bottom 2 cards
    top_row = st.columns(3)
    bottom_row = st.columns([1, 1, 1])  # Change this line

    for i, member in enumerate(team_members):
        if i < 3:
            column = top_row[i]
        else:
            column = bottom_row[i - 3]
        
        with column:
            st.markdown(
            f"""
            <div style="background-color: #1c1d24; border-radius: 10px; padding: 15px; margin-bottom: 10px; text-align: center; max-width: 380px;">
                <img src="{member['image']}" style="border-radius: 50%; width: 150px; height: 150px; object-fit: cover; margin-bottom: 10px;">
                <h3 style="margin: 0; color: #FFFFFF; padding: 0;">{member['name']}</h3>
                <p style="margin: 5px 0; color: #4CAF50;">{member['role']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


elif selected_menu == "Data Overview":
    st.header("1. Data Overview")

    st.subheader("Movies Dataset")
    st.write(movies.head())
    st.write(f"**Total Movies:** {movies.shape[0]:,}")
    st.markdown(f"The Movies dataset contains **{movies.shape[0]:,}** entries. Each entry includes the movie's title and associated genres, providing a comprehensive view of the movie offerings.")
    
    st.subheader("Ratings Dataset (Sampled)")
    st.write(ratings.head())
    st.write(f"**Total Ratings (Sampled):** {ratings.shape[0]:,}")
    st.markdown(f"The Ratings dataset comprises **{ratings.shape[0]:,}** sampled ratings from users. Each rating reflects how much a user liked a particular movie, ranging from 0.5 to 5.0 stars.")
    
    st.subheader("Tags Dataset (Sampled)")
    st.write(tags.head())
    st.write(f"**Total Tags (Sampled):** {tags.shape[0]:,}")
    st.markdown(f"The Tags dataset includes **{tags.shape[0]:,}** sampled tags assigned by users to movies. These tags provide insights into user sentiments and descriptive keywords associated with movies.")

    st.markdown("---")


elif selected_menu == "Data Preprocessing":
    st.header("2. Data Preprocessing")
    
    st.subheader("Handling Missing Values")
    st.write("Checking for missing values in each dataset:")
    missing_movies = movies.isnull().sum()
    missing_ratings = ratings.isnull().sum()
    missing_tags = tags.isnull().sum()
    
    st.write("**Movies Dataset Missing Values:**")
    st.write(missing_movies)
    st.write("**Ratings Dataset Missing Values:**")
    st.write(missing_ratings)
    st.write("**Tags Dataset Missing Values:**")
    st.write(missing_tags)
    
    st.markdown("After thorough inspection, no significant missing values were detected across the datasets. This ensures that our analysis is based on complete and reliable data.")
    
    st.subheader("Converting Timestamps to Readable Dates")
    st.write("Converted `timestamp` to `date` in Ratings and Tags datasets.")
    st.write(ratings[['timestamp', 'date']].head())
    st.write(tags[['timestamp', 'date']].head())
    
    st.markdown("Transforming Unix timestamps into readable dates allows us to analyze temporal trends, such as how ratings and tags evolve over the years.")
    
    st.subheader("Extracting Genres")
    st.write("Extracted and split genres into lists.")
    st.write(movies[['movieId', 'genres']].head())
    
    st.markdown("Splitting genres into individual lists enables one-hot encoding, which is essential for performing association rule mining based on genre combinations.")
    
    st.markdown("---")

elif selected_menu == "Exploratory Analysis":
    st.header("3. Exploratory Data Analysis")
    
    # Filter ratings for years 2005 to 2015
    ratings['year'] = ratings['date'].dt.year
    filtered_ratings = ratings[(ratings['year'] >= 2005) & (ratings['year'] <= 2015)]
    
    st.subheader("Top 10 Most Rated Movies")
    top_movies = filtered_ratings['movieId'].value_counts().head(10).reset_index()
    top_movies.columns = ['movieId', 'ratings_count']
    top_movies = top_movies.merge(movies, on='movieId')
    fig1 = px.bar(top_movies, x='title', y='ratings_count', 
                  title='Top 10 Most Rated Movies (2005-2015)',
                  labels={'title':'Movie Title', 'ratings_count':'Number of Ratings'},
                  hover_data={'title': True, 'ratings_count': True})
    st.plotly_chart(fig1, use_container_width=True)
    
    # Dynamic Explanation for Top 10 Movies
    st.markdown(explain_top_movies(top_movies))
    
    st.subheader("Distribution of Ratings")
    fig2 = px.histogram(filtered_ratings, x='rating', nbins=10, 
                       title='Distribution of Ratings (2005-2015)',
                       labels={'rating':'Rating'},
                       opacity=0.75)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Dynamic Explanation for Ratings Distribution
    st.markdown(explain_ratings_distribution(filtered_ratings))
    
    st.subheader("Ratings Over Time")
    ratings_per_year = filtered_ratings.groupby('year').size().reset_index(name='ratings_count')
    fig3 = px.line(ratings_per_year, x='year', y='ratings_count', 
                   title='Number of Ratings Over Years (2005-2015)',
                   labels={'year':'Year', 'ratings_count':'Number of Ratings'},
                   markers=True)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Dynamic Explanation for Ratings Over Time
    st.markdown(explain_ratings_over_time(ratings_per_year))
    
    st.subheader("Genre Distribution")
    fig4 = px.pie(genre_counts, names='genre', values='count', 
                 title='Genre Distribution (2005-2015)',
                 hover_data=['count'],
                 labels={'count':'Number of Movies'},
                 hole=0.3)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Dynamic Explanation for Genre Distribution
    st.markdown(explain_genre_distribution(genre_counts))
    
    st.markdown("---")


elif selected_menu == "Association Rule Mining":
    st.header("4. Association Rule Mining with Apriori")
    
    st.subheader("Preparing Data for Apriori")
    st.write("Generating frequent itemsets based on movie genres.")
    
    st.write("Genres one-hot encoded.")
    
    st.subheader("Applying Apriori Algorithm")
    st.write("Adjust the minimum support and confidence to generate association rules.")
    
    # Sliders for support and confidence
    min_support = st.slider("Minimum Support", 0.001, 0.01, 0.005, 0.001, key='support_slider_assoc')
    min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05, key='confidence_slider_assoc')
    
    with st.spinner("Running Apriori algorithm..."):
        frequent_itemsets = apriori(genres_onehot, min_support=min_support, use_colnames=True)
    
    st.write(f"**Number of frequent itemsets:** {frequent_itemsets.shape[0]:,}")
    st.markdown(f"We have identified **{frequent_itemsets.shape[0]:,}** frequent genre combinations that appear in at least **{min_support * 100:.2f}%** of the sampled movies.")
    
    st.subheader("Generating Association Rules")
    with st.spinner("Generating association rules..."):
        if not frequent_itemsets.empty:
            try:
                # Calculate num_itemsets as the number of transactions
                num_itemsets = genres_onehot.shape[0]
                
                rules = association_rules(frequent_itemsets, num_itemsets=num_itemsets, metric="confidence", min_threshold=min_confidence)
                
                # Save rules to session state
                st.session_state["rules"] = rules
                st.write(f"**Number of association rules:** {rules.shape[0]:,}")
            except TypeError as e:
                st.error(f"Error generating association rules: {e}")
                st.write("Please ensure that `mlxtend` is correctly installed and updated.")
        else:
            st.warning("No frequent itemsets found. Try lowering the minimum support.")
    
    # Check if rules are not empty and visualize them
    if "rules" in st.session_state and not st.session_state["rules"].empty:
        rules = st.session_state["rules"]
        
        # Convert frozenset to string for Plotly visualization
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))) if isinstance(x, frozenset) else x)
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))) if isinstance(x, frozenset) else x)

        st.markdown("### Top 10 Association Rules")
        top_10_rules = rules.sort_values(by='lift', ascending=False).head(10)
        st.write(top_10_rules)
        
        # Dynamic Explanation for Top 10 Rules
        if not top_10_rules.empty:
            first_rule = top_10_rules.iloc[0]
            st.markdown(explain_top_rule(first_rule))
        
        st.markdown("### Support vs Confidence of Association Rules")
        fig5 = px.scatter(rules, x='support', y='confidence', size='lift',
                          color='lift', hover_data=['antecedents_str', 'consequents_str'],
                          title='Support vs Confidence of Association Rules',
                          labels={'support':'Support', 'confidence':'Confidence', 'lift':'Lift'},
                          size_max=15)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Dynamic Explanation for Scatter Plot
        st.markdown(explain_scatter_plot(rules))
        
        st.markdown("### Interactive Rules Table")
        st.dataframe(rules.sort_values(by='lift', ascending=False).head(50))
        
        # Dynamic Explanation for Interactive Table
        st.markdown(explain_association_table())
    else:
        st.warning("No association rules to display.")


elif selected_menu == "Recommendations":
    st.header("5. Movie Recommendations Based on Association Rules")
    
    # Introduce a Narrative
    st.markdown("""
    ### Empowering Movie Recommendations with Data-Driven Insights

    Movie streaming platforms and theaters face a common challenge: recommending films that truly resonate with users, thereby increasing engagement and satisfaction. With a vast array of movies available, it can be difficult to consistently suggest content that users will enjoy. This is where **Association Rule Mining** comes in.

    By analyzing patterns in genre preferences, we can uncover which genres tend to be enjoyed together. This analysis provides actionable insights that benefit various stakeholders:
    """)

    # Check if the rules exist in the session state
    if "rules" in st.session_state and not st.session_state["rules"].empty:
        rules = st.session_state["rules"]  # Access the rules from session state
        
        st.markdown("### Select a Genre for Recommendations")
        selected_genre = st.selectbox("Choose a Genre", sorted(genre_counts['genre'].unique()))
        
        st.markdown("### Top Associated Genres")
        
        # Convert 'antecedents_str' back to sets for accurate filtering
        rules['antecedents_set'] = rules['antecedents_str'].apply(lambda x: set(x.split(", ")) if isinstance(x, str) else x)
        
        # Filter based on the selected genre
        filtered_rules = rules[rules['antecedents_set'].apply(lambda x: selected_genre in x)]
        
        if not filtered_rules.empty:
            top_associations = filtered_rules.sort_values(by='lift', ascending=False).head(10)
            
            # Extract and display unique associated genres
            unique_associations = top_associations['consequents_str'].str.split(", ").explode().drop_duplicates()
            
            # Combine the associated genres into a single paragraph
            association_paragraph = ', '.join(unique_associations.sort_values())
            
            st.markdown(f"### Users who like **{selected_genre}** also like: {association_paragraph}.")
            
            # Combine all practical applications, solving challenges, and empowering decisions into one paragraph
            st.markdown(
                """
                To address the common challenges faced by movie streaming platforms and theaters in recommending content that truly resonates with users, leveraging **Association Rule Mining** can enhance recommendations by identifying genre associations. For movie streaming services, these insights can refine algorithms by suggesting complementary genres like **Children** and **Animation** to users who enjoy **Adventure**, boosting satisfaction and engagement. Similarly, movie theaters can curate diverse lineups by pairing genres such as **Adventure** and **Children**, attracting a wider audience. Individual users also benefit by discovering new genres—such as **Children** or **Animation**—that align with their tastes, enriching their viewing experience. These data-driven strategies not only personalize recommendations but also empower stakeholders to foster greater engagement, loyalty, and platform usage, ensuring that both businesses and users enjoy a more tailored and satisfying movie experience.
                """
            )
            
            st.markdown("### Visualization of Associations")
            # Create a bar chart with associated genres and their lift values
            fig6 = px.bar(top_associations, 
                         x='consequents_str', 
                         y='lift',
                         hover_data=['support', 'confidence'],
                         title=f'Top Associated Genres with {selected_genre}',
                         labels={'lift':'Lift', 'consequents_str':'Associated Genre'},
                         text='lift')
            fig6.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig6.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig6, use_container_width=True)
            
            # Dynamic Explanation for Recommendations Bar Chart
            st.markdown(explain_recommendations(selected_genre, top_associations))
        else:
            st.markdown(f"No association rules found for the selected genre: **{selected_genre}**.")
    else:
        st.warning("Association rules have not been generated yet. Please go to the 'Association Rule Mining' section first.")
    
    st.markdown("---")
    
    # Concluding the Journey
    st.markdown("### Concluding Our Journey")
    st.markdown(
        """
        Through meticulous analysis and data-driven methodologies, we've uncovered meaningful associations within the MovieLens 20M dataset. These insights not only enhance movie recommendation systems but also provide strategic guidance for stakeholders aiming to elevate user satisfaction and engagement.

        **Key Takeaways:**
        - **Personalized Recommendations:** Leveraging genre associations to tailor movie suggestions aligns closely with user preferences, fostering a more engaging viewing experience.
        - **Strategic Content Curation:** For movie streaming platforms and theaters, understanding genre pairings aids in curating content that appeals to a broader audience.
        - **Enhanced User Experience:** Individual users benefit from discovering new genres that complement their existing tastes, enriching their personal movie libraries.
        
        Embracing these data-driven strategies ensures that both businesses and users can enjoy a more tailored and satisfying movie experience.
        """
    )
    
    # Closing Animation
    if lottie_closing:
        st_lottie(lottie_closing, height=200, key="closing")
    else:
        st.error("Failed to load the closing animation.")
    
    # Footer
    st.markdown(
        """
        ---
        <div style="text-align: center;">
            <strong>Dataset:</strong> <a href="https://grouplens.org/datasets/movielens/20m/" style="color: #4CAF50;">MovieLens 20M</a> | 
            <strong>Project Report:</strong> Final Project Report by Group Nime | 
            <strong>Developed with:</strong> Streamlit, Python, Pandas, MLxtend, Plotly
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------
# Footer (Optional)
# --------------------------
# Note: Since the footer is included in the "Concluding Our Journey" section, it's not necessary to add it again here.
