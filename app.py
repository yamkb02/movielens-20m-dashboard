import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from scipy.stats import skew
import seaborn as sns

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




def extract_unique_genres(top_associations):
    """
    Extracts unique genres from the top_associations DataFrame,
    ensuring each genre appears only once with its highest lift value.
    """
    print("Extracting unique genres with highest lift values.")
    
    # Initialize a dictionary to hold the highest lift per genre
    genre_lift_dict = {}
    
    for _, row in top_associations.iterrows():
        # Split the consequents_str into individual genres
        genres = [genre.strip() for genre in row['consequents_str'].split(',')]
        for genre in genres:
            # Update the lift value if it's higher than the existing one
            if genre not in genre_lift_dict or row['lift'] > genre_lift_dict[genre]:
                genre_lift_dict[genre] = row['lift']
                print(f"Updated lift for genre '{genre}': {row['lift']:.2f}")
    
    # Convert the dictionary to a sorted list of tuples (genre, lift)
    sorted_genres = sorted(genre_lift_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"Unique genres sorted by lift: {sorted_genres}")
    
    return sorted_genres

def explain_recommendations(selected_genre, top_associations, genre_colors):
    if not top_associations.empty:
        print("Step 2: Getting unique associations and their highest lift values.")
        
        # Step 2: Extract unique genres with highest lift values
        unique_genres = extract_unique_genres(top_associations)
        
        # Step 3: Format the unique genres with lift values and HTML colors
        print("Step 3: Formatting the unique genres with lift values.")
        associations_list = [
            f"<strong><span style=\"color:{genre_colors.get(genre, '#000000')};\">{genre}</span></strong> (lift: {lift:.2f})"
            for genre, lift in unique_genres
        ]
        print(f"Formatted associations list with lift values: {associations_list}")
        
        # Combine the associations into a single paragraph
        print("Step 4: Combining the associations into a single paragraph.")
        if len(associations_list) > 1:
            associations_paragraph = ', '.join(associations_list[:-1]) + f", and {associations_list[-1]}."
        else:
            associations_paragraph = associations_list[0]
        print(f"Final associations paragraph: {associations_paragraph}")
        
        # Start the explanation text
        print("Step 5: Creating the explanation text.")
        explanation = f"### In-Depth Look at Recommendations for **{colorize_genre_string(selected_genre, genre_colors)}**\n\n"
        explanation += (
            f"The bar chart above highlights the top genres associated with **{colorize_genre_string(selected_genre, genre_colors)}**. "
            f"Each bar represents an associated genre and its **lift** value, which indicates the strength of the association. "
            f"These associations include {associations_paragraph} "
            f"Users who enjoy **{colorize_genre_string(selected_genre, genre_colors)}** movies often find these genres appealing, "
            f"enhancing their viewing experience with complementary content."
        )
        
        # Consolidate Practical Applications, Solving Challenges, and Empowering Decisions into one paragraph
        print("Step 6: Adding practical applications to the explanation text.")
        recommendation_genres = [genre for genre, _ in unique_genres]
        recommendation_genres_colored = [colorize_genre_string(genre, genre_colors) for genre in recommendation_genres]
        recommendation_genres_str = ', '.join(recommendation_genres_colored)
        
        explanation += "\n\n"
        explanation += (
            "Movie streaming platforms and theaters often struggle to recommend content that truly resonates with users. "
            "By leveraging **Association Rule Mining**, genre associations can be identified to improve recommendations. "
            f"For instance, streaming services can refine their algorithms by suggesting complementary genres like **{recommendation_genres_str}** to users who enjoy **{colorize_genre_string(selected_genre, genre_colors)}**, increasing user satisfaction. "
            f"Similarly, theaters can curate more appealing lineups by pairing genres such as **{colorize_genre_string(selected_genre, genre_colors)}** with **{recommendation_genres_str}**, attracting a wider audience. "
            "These data-driven strategies personalize recommendations, foster greater engagement, and promote platform loyalty, ensuring that both businesses and users benefit from a more tailored movie experience."
        )
        
        print("Step 7: Final explanation generated.")
        return explanation
    else:
        print("No strong associations found for the selected genre.")
        return "No strong associations found for the selected genre."

    

# Color genres
# Load movies.csv file
movies_df = pd.read_csv("ml-20m/movies.csv")

# Step 1: Extract the genres and split them into individual genres
genres_series = movies_df['genres'].str.split('|').explode().unique()

# Step 2: Generate a color mapping for each genre
def generate_genre_colors(genres):
    genre_colors = {}
    # Use seaborn's color palette to generate distinct colors
    color_palette = sns.color_palette("Set3", len(genres))
    
    # Map each genre to a unique color from the palette
    for i, genre in enumerate(genres):
        genre_colors[genre] = f'#{int(color_palette[i][0] * 255):02x}{int(color_palette[i][1] * 255):02x}{int(color_palette[i][2] * 255):02x}'  # RGB to Hex
    
    return genre_colors

# Get the unique genres from the dataset
unique_genres = genres_series.tolist()

# Generate the color mapping for genres
genre_colors = generate_genre_colors(unique_genres)

# Function to colorize genre strings dynamically
def colorize_genre_string(genre_string, genre_colors):
    genres = genre_string.split(', ')
    colored_genres = []
    for genre in genres:
        genre_color = genre_colors.get(genre, "#000000")  # Default to black if no color is found
        colored_genres.append(f'<span style="color:{genre_color};">{genre}</span>')
    return ', '.join(colored_genres)

# Function to render the HTML table with genre colors
def render_html_table(df):
    html = df.to_html(escape=False)  # Escape=False allows raw HTML in the table
    st.markdown(html, unsafe_allow_html=True)

# Now when applying this colorization logic, it will give you distinct colors





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
        padding-right: 2rem;
        padding-left: 2rem;
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
        text-align: center;
    }
    .card h3 {
        color: #4CAF50;
        margin-bottom: 10px;
    }
    /* Slider Styling */
    .custom-slider {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .custom-slider .stSlider > div {
        width: 50%;
    }
    /* Team Member Image Styling */
    .team-member img {
        border-radius: 50%;
        width: 150px;
        height: 150px;
        object-fit: cover;
        margin-bottom: 10px;
    }
    /* Footer Styling */
    .footer {
        text-align: center;
        margin-top: 20px;
        color: #4CAF50;
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

    # Team member cards (without images)
    team_members = [
        {"name": "Mark Kenneth Badilla", "role": "Leader", "years": "SY 2020-2023"},
        {"name": "Rob Borinaga", "role": "Member", "years": "SY 2019-2023"},
        {"name": "Alestair Cyril Coyoca", "role": "Member", "years": "SY 2021-2024"},
        {"name": "Carmelyn Nime Gerali", "role": "Member", "years": "SY 2018-2022"},
        {"name": "James Alein Ocampo", "role": "Member", "years": "SY 2020-2024"}
    ]

    # Create two rows: one for the top 3 cards, another for the bottom 2 cards
    top_row = st.columns(3)
    bottom_row = st.columns([1, 1, 1])

    for i, member in enumerate(team_members):
        if i < 3:
            column = top_row[i]
        else:
            column = bottom_row[i - 3]
        
        with column:
            st.markdown(
            f"""
            <div style="background-color: #1c1d24; border-radius: 10px; padding: 15px; margin-bottom: 10px; text-align: center; max-width: 380px;">
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
        
        # Convert frozenset to string for easier visualization
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))) if isinstance(x, frozenset) else x)
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))) if isinstance(x, frozenset) else x)

        # Apply colorization to antecedents and consequents
        rules['antecedents_colored'] = rules['antecedents_str'].apply(lambda x: colorize_genre_string(x, genre_colors))
        rules['consequents_colored'] = rules['consequents_str'].apply(lambda x: colorize_genre_string(x, genre_colors))

        # Ensure antecedents_str and consequents_str are included in the selected columns
        selected_columns = ['antecedents_str', 'consequents_str', 'antecedents_colored', 'consequents_colored', 'support', 'confidence', 'lift']
        
        st.markdown("### Top 10 Association Rules")
        top_10_rules = rules[selected_columns].sort_values(by='lift', ascending=False).head(10)

        # Render the top 10 rules as an HTML table with colored genres
        render_html_table(top_10_rules[['antecedents_colored', 'consequents_colored', 'support', 'confidence', 'lift']])
        
        # Dynamic Explanation for Top 10 Rules
        if not top_10_rules.empty:
            # Pass the first rule row as a series
            first_rule = top_10_rules.iloc[0]
            st.markdown(explain_top_rule(first_rule))

        
        # Scatter plot of Support vs Confidence
        st.markdown("### Support vs Confidence of Association Rules")
        fig5 = px.scatter(rules, x='support', y='confidence', size='lift',
                        color='lift', hover_data=['antecedents_colored', 'consequents_colored'],
                        title='Support vs Confidence of Association Rules',
                        labels={'support':'Support', 'confidence':'Confidence', 'lift':'Lift'},
                        size_max=15)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Dynamic Explanation for Scatter Plot
        st.markdown(explain_scatter_plot(rules))
        
        # Interactive Rules Table (HTML)
        st.markdown("### Interactive Rules Table")
        # Render the entire table as HTML
        render_html_table(rules[['antecedents_colored', 'consequents_colored', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(50))
        
        # Dynamic Explanation for Interactive Table
        st.markdown(explain_association_table())
    else:
        st.warning("No association rules to display.")



elif selected_menu == "Recommendations":
    st.header("5. Movie Recommendations Based on Association Rules")
    
    # Introduce a Narrative
    st.markdown(""" ### Empowering Movie Recommendations with Data-Driven Insights
    Movie streaming platforms and theaters face a common challenge: recommending films that truly resonate with users, thereby increasing engagement and satisfaction. With a vast array of movies available, it can be difficult to consistently suggest content that users will enjoy. This is where **Association Rule Mining** comes in.
    By analyzing patterns in genre preferences, we can uncover which genres tend to be enjoyed together. This analysis provides actionable insights that benefit various stakeholders: """)
    
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
            
            # Colorize the associated genres and create a paragraph
            association_paragraph = ', '.join([colorize_genre_string(genre, genre_colors) for genre in unique_associations.sort_values()])
            
            # Render the paragraph with colored genres
            st.markdown(f"### Users who like **{colorize_genre_string(selected_genre, genre_colors)}** also like: {association_paragraph}.", unsafe_allow_html=True)
            
            # Combine all practical applications, solving challenges, and empowering decisions into one paragraph
            st.markdown(
                f"""
                To address the common challenges faced by movie streaming platforms and theaters in recommending content that truly resonates with users, leveraging **Association Rule Mining** can enhance recommendations by identifying genre associations. For movie streaming services, these insights can refine algorithms by suggesting complementary genres like **{', '.join([colorize_genre_string(genre, genre_colors) for genre in unique_associations])}** to users who enjoy **{colorize_genre_string(selected_genre, genre_colors)}**, boosting satisfaction and engagement. Similarly, movie theaters can curate diverse lineups by pairing genres such as **{colorize_genre_string(selected_genre, genre_colors)}** and **{', '.join([colorize_genre_string(genre, genre_colors) for genre in unique_associations])}**, attracting a wider audience. Individual users also benefit from discovering new genres—such as **{', '.join([colorize_genre_string(genre, genre_colors) for genre in unique_associations])}**—that align with their tastes, enriching their personal movie libraries. These data-driven strategies not only personalize recommendations but also empower stakeholders to foster greater engagement, loyalty, and platform usage, ensuring that both businesses and users enjoy a more tailored and satisfying movie experience.
                """, unsafe_allow_html=True
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
            
            # Update the colors of the bars
            fig6.update_traces(marker_color=[genre_colors.get(genre, "#000000") for genre in top_associations['consequents_str']])
            
            st.plotly_chart(fig6, use_container_width=True)
            
            # Dynamic Explanation for Recommendations Bar Chart
            st.markdown(explain_recommendations(selected_genre, top_associations, genre_colors), unsafe_allow_html=True)
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
        """, unsafe_allow_html=True
    )
    )
