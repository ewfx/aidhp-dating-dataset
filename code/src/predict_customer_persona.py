import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import os

class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class PredictCustomerPersona:
    def __init__(self, kmeans_model=None, scaler=None, persona_names=None, feature_columns=None):
        """Initialize the predictor with optional pre-trained model"""
        self.kmeans = kmeans_model
        self.scaler = scaler
        self.persona_names = persona_names
        self.feature_columns = feature_columns
        
    def fit(self, data):
        """Fit the model with the provided data"""
        self.data = data
        return self
        
    def predict_customer(self, customer_data):
        """Predict persona for a single customer with confidence scores"""
        # Extract features
        features = customer_data[self.feature_columns].values.reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Get distances to all cluster centers
        distances = self.kmeans.transform(scaled_features)
        
        # Convert distances to probabilities using softmax
        probabilities = self._distance_to_probability(distances[0])
        
        # Create predictions with confidence scores
        predictions = []
        for cluster_id, prob in enumerate(probabilities):
            predictions.append({
                'cluster_id': int(cluster_id),
                'persona_name': self.persona_names[cluster_id],
                'confidence_score': float(prob)
            })
        
        # Sort by confidence score
        predictions.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return predictions
        
    def predict_all_customers(self):
        """Predict personas for all customers in the dataset"""
        predictions = []
        
        # Scale all features
        scaled_features = self.scaler.transform(self.data[self.feature_columns])
        
        # Calculate distances to all cluster centers for all customers
        distances = self.kmeans.transform(scaled_features)
        
        # Create predictions for each customer
        for idx, row in self.data.iterrows():
            # Get probabilities for each persona
            probabilities = self._distance_to_probability(distances[idx])
            
            # Create persona predictions with confidence scores
            persona_predictions = []
            for cluster_id in range(len(self.persona_names)):
                persona_predictions.append({
                    'cluster_id': int(cluster_id),
                    'persona_name': self.persona_names[cluster_id],
                    'confidence_score': float(probabilities[cluster_id])
                })
            
            # Sort predictions by confidence score in descending order
            persona_predictions.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            prediction = {
                'customer_ID': row['customer_ID'],
                'persona_predictions': persona_predictions,
                'user_features': {
                    'Age': row['Age'],
                    'Income': row['Income per Year (in dollars)'],
                    'avg_transaction': row['avg_transaction_amount'],
                    'total_spend': row['total_spend'],
                    'transaction_count': row['transaction_count'],
                    'avg_sentiment': row['avg_sentiment'],
                    'unique_intents': row['unique_intents']
                }
            }
            predictions.append(prediction)
        
        # Save predictions to files
        self._save_predictions(predictions)
        
        return predictions
    
    def _distance_to_probability(self, distances):
        """Convert distances to probabilities using softmax"""
        # Convert distances to similarities (negative distances)
        similarities = -distances
        
        # Apply softmax
        exp_similarities = np.exp(similarities - np.max(similarities))
        probabilities = exp_similarities / exp_similarities.sum()
        
        return probabilities
    
    def _save_predictions(self, predictions):
        """Save predictions to JSON and CSV files"""
        # Create artifacts directory if it doesn't exist
        artifacts_dir = '../artifacts'
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save detailed predictions to JSON
        output_json = os.path.join(artifacts_dir, 'persona_predictions.json')
        with open(output_json, 'w') as f:
            json.dump(predictions, f, indent=4, cls=NumpyEncoder)
        
        # Create DataFrame for CSV with top 2 personas for each customer
        csv_data = []
        for pred in predictions:
            top_personas = pred['persona_predictions'][:2]  # Get top 2 personas
            row = {
                'customer_ID': pred['customer_ID'],
                'primary_persona': top_personas[0]['persona_name'],
                'primary_confidence': top_personas[0]['confidence_score'],
                'secondary_persona': top_personas[1]['persona_name'],
                'secondary_confidence': top_personas[1]['confidence_score'],
                **pred['user_features']
            }
            csv_data.append(row)
        
        # Save predictions to CSV
        output_csv = os.path.join(artifacts_dir, 'persona_predictions.csv')
        pd.DataFrame(csv_data).to_csv(output_csv, index=False)

if __name__ == "__main__":
    # This section is for testing the predictor independently
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    profiles_df = pd.read_csv('../resources/customer_profiles.csv')
    transactions_df = pd.read_csv('../resources/customer_transaction_history.csv')
    sentiments_df = pd.read_csv('../resources/customer_social_sentiments.csv')
    
    # Calculate metrics
    transaction_metrics = transactions_df.groupby('customer_ID').agg({
        'Amt (in dollars)': ['mean', 'sum', 'count']
    }).reset_index()
    transaction_metrics.columns = ['customer_ID', 'avg_transaction_amount', 'total_spend', 'transaction_count']
    
    sentiment_metrics = sentiments_df.groupby('customer_ID').agg({
        'Sentiment Score': 'mean',
        'Intent': 'nunique'
    }).reset_index()
    sentiment_metrics.columns = ['customer_ID', 'avg_sentiment', 'unique_intents']
    
    # Merge data
    final_df = profiles_df.merge(transaction_metrics, on='customer_ID', how='left')
    final_df = final_df.merge(sentiment_metrics, on='customer_ID', how='left')
    
    # Fill missing values
    numeric_columns = ['avg_transaction_amount', 'total_spend', 'transaction_count', 'avg_sentiment', 'unique_intents']
    final_df[numeric_columns] = final_df[numeric_columns].fillna(final_df[numeric_columns].mean())
    
    # Define features
    feature_columns = [
        'Age',
        'Income per Year (in dollars)',
        'avg_transaction_amount',
        'total_spend',
        'transaction_count',
        'avg_sentiment',
        'unique_intents'
    ]
    
    # Initialize models
    kmeans = KMeans(n_clusters=5, random_state=42)
    scaler = StandardScaler()
    
    # Scale features and fit KMeans
    features = final_df[feature_columns].values
    scaled_features = scaler.fit_transform(features)
    kmeans.fit(scaled_features)
    
    # Define persona names
    persona_names = [
        "High-Value Real Estate Investors",
        "Established Professionals",
        "Young Professionals",
        "High-Income Middle-Aged Professionals",
        "Active Middle-Aged Professionals"
    ]
    
    # Create predictor
    predictor = PredictCustomerPersona(
        kmeans_model=kmeans,
        scaler=scaler,
        persona_names=persona_names,
        feature_columns=feature_columns
    )
    
    # Fit and predict
    predictor.fit(final_df)
    predictions = predictor.predict_all_customers()
    
    # Print example predictions
    print("\nExample Predictions:")
    print("=" * 50)
    for pred in predictions[:3]:  # Show first 3 predictions
        print(f"\nCustomer ID: {pred['customer_ID']}")
        print("Persona Predictions:")
        for p in pred['persona_predictions'][:2]:  # Show top 2 personas
            print(f"- {p['persona_name']}: {p['confidence_score']:.2%} confidence") 