from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

data = pd.read_csv('./data/cleaned_recipes.csv')

nutritional_columns = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                       'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']

def last_order(order_name):
    """
    Fetch nutritional data for the given last ordered food.

    Args:
        order_name (str): The name of the last ordered food.

    Returns:
        pd.Series or None: Nutritional information for the food, or None if not found.
    """
    match = data[data['Name'] == order_name]
    if not match.empty:
        return match[nutritional_columns].iloc[0]
    else:
        print(f"'{order_name}' not found in the data.")
        return None

def food_recommender(input_food, user_allergens=[]):
    """
    Recommends food based on nutritional similarity to a given food, while filtering out allergens.

    Args:
        input_food (str): Name of the input food.
        user_allergens (list): List of allergens to filter out.

    Returns:
        pd.DataFrame: Recommendations with relevant details.
    """
    input_nutritional_data = last_order(input_food)
    if input_nutritional_data is None:
        return None  # Exit if no matching food found.

    scaler = MinMaxScaler()
    input_food_scaled = scaler.fit_transform([input_nutritional_data.values])  # Scale input food data

    # Apply allergen filter
    if user_allergens:
        filtered_data = data[~data['Allergens'].str.contains('|'.join(user_allergens), na=False)]
    else:
        filtered_data = data

    # Scale nutritional data of the filtered dataset
    filtered_food_data = scaler.transform(filtered_data[nutritional_columns])

    # Calculate similarity scores
    similarity_scores = cosine_similarity(input_food_scaled, filtered_food_data)

    # Get top recommendations
    top_indices = similarity_scores[0].argsort()[::-1][:10]
    recommendations = filtered_data.iloc[top_indices]

    # Relevant columns for display
    relevant_columns = ['Name', 'Ingredients', 'Allergens'] + nutritional_columns
    return recommendations[relevant_columns].to_dict(orient='records')
