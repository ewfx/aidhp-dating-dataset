import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
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

class CreateCustomerPersonas:
    def __init__(self, n_clusters=5):
        """Initialize the customer persona creator"""
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.persona_names = [
            "High-Value Real Estate Investors",
            "Established Professionals",
            "Young Professionals",
            "High-Income Middle-Aged Professionals",
            "Active Middle-Aged Professionals"
        ]
        
    def load_data(self):
        """Load and prepare the data"""
        # Load the data
        self.profiles_df = pd.read_csv('../resources/customer_profiles.csv')
        self.transactions_df = pd.read_csv('../resources/customer_transaction_history.csv')
        self.sentiments_df = pd.read_csv('../resources/customer_social_sentiments.csv')
        
    def preprocess_data(self):
        """Preprocess and combine data from all sources"""
        # Calculate transaction metrics
        transaction_metrics = self.transactions_df.groupby('customer_ID').agg({
            'Amt (in dollars)': ['mean', 'sum', 'count']
        }).reset_index()
        transaction_metrics.columns = ['customer_ID', 'avg_transaction_amount', 'total_spend', 'transaction_count']
        
        # Calculate sentiment metrics
        sentiment_metrics = self.sentiments_df.groupby('customer_ID').agg({
            'Sentiment Score': 'mean',
            'Intent': 'nunique'
        }).reset_index()
        sentiment_metrics.columns = ['customer_ID', 'avg_sentiment', 'unique_intents']
        
        # Merge all metrics with profiles
        self.final_df = self.profiles_df.merge(transaction_metrics, on='customer_ID', how='left')
        self.final_df = self.final_df.merge(sentiment_metrics, on='customer_ID', how='left')
        
        # Fill missing values with mean
        numeric_columns = ['avg_transaction_amount', 'total_spend', 'transaction_count', 'avg_sentiment', 'unique_intents']
        self.final_df[numeric_columns] = self.final_df[numeric_columns].fillna(self.final_df[numeric_columns].mean())
        
    def prepare_features(self):
        """Prepare features for clustering"""
        # Select features for clustering
        feature_columns = [
            'Age',
            'Income per Year (in dollars)',
            'avg_transaction_amount',
            'total_spend',
            'transaction_count',
            'avg_sentiment',
            'unique_intents'
        ]
        
        # Scale features
        features = self.scaler.fit_transform(self.final_df[feature_columns])
        return features
        
    def perform_clustering(self):
        """Perform clustering on the prepared features"""
        # Prepare features
        features = self.prepare_features()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(scaled_features)
        
        return cluster_labels
        
    def create_personas(self):
        """Create customer personas using clustering"""
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Perform clustering
        cluster_labels = self.perform_clustering()
        
        # Create cluster results DataFrame
        cluster_results = pd.DataFrame({
            'customer_ID': self.final_df['customer_ID'],
            'cluster': cluster_labels,
            'persona': [self.persona_names[label] for label in cluster_labels]
        })
        
        # Create artifacts directory if it doesn't exist
        artifacts_dir = '../artifacts'
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save cluster labels
        cluster_results.to_csv(os.path.join(artifacts_dir, 'cluster_labels.csv'), index=False)
        
        # Generate insights for each cluster
        cluster_insights = self.generate_cluster_insights(cluster_labels)
        
        # Save insights to JSON
        with open(os.path.join(artifacts_dir, 'cluster_insights.json'), 'w') as f:
            json.dump(cluster_insights, f, indent=4, cls=NumpyEncoder)
        
        # Predict personas for all customers
        from predict_customer_persona import PredictCustomerPersona
        predictor = PredictCustomerPersona(
            kmeans_model=self.kmeans,
            scaler=self.scaler,
            persona_names=self.persona_names,
            feature_columns=self.get_feature_columns()
        )
        predictor.fit(self.final_df)
        predictions = predictor.predict_all_customers()
        
        return cluster_insights, predictions
    
    def get_feature_columns(self):
        """Get the list of feature columns used for clustering"""
        return [
            'Age',
            'Income per Year (in dollars)',
            'avg_transaction_amount',
            'total_spend',
            'transaction_count',
            'avg_sentiment',
            'unique_intents'
        ]
    
    def analyze_clusters(self):
        """Analyze the characteristics of each cluster"""
        cluster_insights = []
        
        for cluster_id in range(self.n_clusters):
            cluster_data = self.final_df[self.final_df['Cluster'] == cluster_id]
            
            # Calculate cluster statistics
            stats = {
                'cluster_id': cluster_id,
                'name': self._get_persona_name(cluster_data),
                'size': len(cluster_data),
                'avg_age': cluster_data['Age'].mean(),
                'avg_income': cluster_data['Income per Year (in dollars)'].mean(),
                'avg_transaction': cluster_data['avg_transaction_amount'].mean(),
                'transaction_freq': cluster_data['transaction_count'].mean(),
                'avg_sentiment': cluster_data['avg_sentiment'].mean(),
                'unique_intents': cluster_data['unique_intents'].mean(),
                'common_interests': self._get_common_interests(cluster_data),
                'common_occupations': cluster_data['Occupation'].value_counts().head(3).index.tolist()
            }
            
            cluster_insights.append(stats)
            
            # Print cluster summary
            print(f"\n{stats['name']} (Cluster {cluster_id})")
            print("-" * 30)
            print(f"Size: {stats['size']} customers")
            print(f"Average Age: {stats['avg_age']:.2f}")
            print(f"Average Income: ${stats['avg_income']:,.2f}")
            print(f"Average Transaction: ${stats['avg_transaction']:.2f}")
            print(f"Transaction Frequency: {stats['transaction_freq']:.2f}")
            print(f"Average Sentiment: {stats['avg_sentiment']:.2f}")
            print(f"Unique Intents: {stats['unique_intents']:.2f}")
            print("Common Interests:", ", ".join(stats['common_interests']))
            print("Common Occupations:", ", ".join(stats['common_occupations']))
        
        self.visualize_clusters()
        self.save_cluster_insights(cluster_insights)
        return cluster_insights
    
    def _get_persona_name(self, cluster_data):
        """Generate a descriptive name for the persona based on its characteristics"""
        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['Income per Year (in dollars)'].mean()
        avg_transaction = cluster_data['avg_transaction_amount'].mean()
        transaction_freq = cluster_data['transaction_count'].mean()
        
        # Determine age group
        if avg_age < 35:
            age_group = "Young"
        elif avg_age < 45:
            age_group = "Middle-Aged"
        else:
            age_group = "Mature"
            
        # Determine income level
        if avg_income > 150000:
            income_level = "High-Income"
        elif avg_income > 120000:
            income_level = "Upper-Middle"
        else:
            income_level = "Middle"
            
        # Determine spending pattern
        if avg_transaction > 50000:
            spending = "High-Value"
        elif transaction_freq > 15:
            spending = "Frequent"
        else:
            spending = "Moderate"
            
        # Determine primary occupation
        primary_occupation = cluster_data['Occupation'].value_counts().index[0]
        
        # Combine characteristics to create name
        if spending == "High-Value":
            if "Real Estate" in cluster_data['Interests'].value_counts().index[0]:
                return "High-Value Real Estate Investors"
            else:
                return f"{income_level} {primary_occupation}s"
        elif age_group == "Young":
            return "Young Professionals"
        elif age_group == "Middle-Aged":
            if avg_income > 150000:
                return "High-Income Middle-Aged Professionals"
            else:
                return "Active Middle-Aged Professionals"
        else:
            return "Established Professionals"
    
    def _get_common_interests(self, cluster_data):
        """Extract common interests from a cluster"""
        all_interests = []
        for interests in cluster_data['Interests'].dropna():
            all_interests.extend([i.strip() for i in interests.split(',')])
        
        interest_counts = pd.Series(all_interests).value_counts()
        return interest_counts.head(3).index.tolist()
    
    def visualize_clusters(self):
        """Create visualizations of the clusters"""
        features = self.prepare_features()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Customer Persona Clusters Analysis', fontsize=16)
        
        # Plot 1: Age vs Income
        sns.scatterplot(
            data=self.final_df,
            x='Age',
            y='Income per Year (in dollars)',
            hue='Cluster',
            ax=axes[0,0]
        )
        axes[0,0].set_title('Age vs Income Distribution')
        
        # Plot 2: Transaction Amount vs Frequency
        sns.scatterplot(
            data=self.final_df,
            x='avg_transaction_amount',
            y='transaction_count',
            hue='Cluster',
            ax=axes[0,1]
        )
        axes[0,1].set_title('Transaction Amount vs Frequency')
        
        # Plot 3: Sentiment vs Engagement
        sns.scatterplot(
            data=self.final_df,
            x='avg_sentiment',
            y='unique_intents',
            hue='Cluster',
            ax=axes[1,0]
        )
        axes[1,0].set_title('Sentiment vs Engagement')
        
        # Plot 4: Total Spend Distribution
        sns.boxplot(
            data=self.final_df,
            x='Cluster',
            y='total_spend',
            ax=axes[1,1]
        )
        axes[1,1].set_title('Total Spend Distribution by Cluster')
        
        plt.tight_layout()
        plt.savefig('code/artifacts/cluster_visualization.png')
        plt.close()
    
    def save_cluster_insights(self, cluster_insights):
        """Save cluster insights in multiple formats"""
        # Save as CSV for spreadsheet access
        insights_df = pd.DataFrame(cluster_insights)
        insights_df.to_csv('code/artifacts/cluster_insights.csv', index=False)
        
        # Save as a readable text file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('code/artifacts/customer_personas.txt', 'w') as f:
            f.write(f"Customer Personas Analysis\n")
            f.write(f"Generated on: {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            for insights in cluster_insights:
                f.write(f"{insights['name']} (Cluster {insights['cluster_id']})\n")
                f.write("-" * 30 + "\n")
                f.write(f"Size: {insights['size']} customers\n")
                f.write(f"Average Age: {insights['avg_age']:.2f}\n")
                f.write(f"Average Income: ${insights['avg_income']:,.2f}\n")
                f.write(f"Average Transaction: ${insights['avg_transaction']:.2f}\n")
                f.write(f"Transaction Frequency: {insights['transaction_freq']:.2f}\n")
                f.write(f"Average Sentiment: {insights['avg_sentiment']:.2f}\n")
                f.write(f"Unique Intents: {insights['unique_intents']:.2f}\n")
                f.write(f"Common Interests: {', '.join(insights['common_interests'])}\n")
                f.write(f"Common Occupations: {', '.join(insights['common_occupations'])}\n")
                f.write("\n")

    def generate_cluster_insights(self, cluster_labels):
        """Generate insights for each cluster"""
        # Add cluster labels to the dataframe
        self.final_df['Cluster'] = cluster_labels
        
        cluster_insights = []
        
        for cluster_id in range(self.n_clusters):
            cluster_data = self.final_df[self.final_df['Cluster'] == cluster_id]
            
            # Calculate cluster statistics
            stats = {
                'cluster_id': cluster_id,
                'name': self.persona_names[cluster_id],
                'size': len(cluster_data),
                'avg_age': float(cluster_data['Age'].mean()),
                'avg_income': float(cluster_data['Income per Year (in dollars)'].mean()),
                'avg_transaction': float(cluster_data['avg_transaction_amount'].mean()),
                'transaction_freq': float(cluster_data['transaction_count'].mean()),
                'avg_sentiment': float(cluster_data['avg_sentiment'].mean()),
                'unique_intents': float(cluster_data['unique_intents'].mean()),
                'common_interests': self._get_common_interests(cluster_data),
                'common_occupations': cluster_data['Occupation'].value_counts().head(3).index.tolist()
            }
            
            cluster_insights.append(stats)
            
            # Print cluster summary
            print(f"\n{stats['name']} (Cluster {cluster_id})")
            print("-" * 30)
            print(f"Size: {stats['size']} customers")
            print(f"Average Age: {stats['avg_age']:.2f}")
            print(f"Average Income: ${stats['avg_income']:,.2f}")
            print(f"Average Transaction: ${stats['avg_transaction']:.2f}")
            print(f"Transaction Frequency: {stats['transaction_freq']:.2f}")
            print(f"Average Sentiment: {stats['avg_sentiment']:.2f}")
            print(f"Unique Intents: {stats['unique_intents']:.2f}")
            print("Common Interests:", ", ".join(stats['common_interests']))
            print("Common Occupations:", ", ".join(stats['common_occupations']))
        
        return cluster_insights

    def run(self, generate_insights=True):
        """Run the complete pipeline"""
        print("Starting customer persona analysis...")
        
        # Create personas and get predictions
        cluster_insights, predictions = self.create_personas()
        
        # Save predictions
        with open('../artifacts/persona_predictions.json', 'w') as f:
            json.dump(predictions, f, indent=4, cls=NumpyEncoder)
        
        print("\nGenerating credit card recommendations...")
        from recommend_credit_cards import CreditCardRecommender
        recommender = CreditCardRecommender()
        recommendations = recommender.recommend_cards(predictions)
        
        # Save recommendations
        with open('../artifacts/credit_card_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=4, cls=NumpyEncoder)
        
        # Comment out email generation to save tokens
        """
        print("\nGenerating personalized ad emails...")
        from generate_ad_emails import AdEmailGenerator
        email_generator = AdEmailGenerator()
        num_emails = email_generator.generate_emails()
        """
        print("\nSkipping email generation to save tokens...")
        num_emails = 0
        
        # Generate AI-powered insights if requested
        if generate_insights:
            print("\nGenerating AI-powered insights for personas...")
            from generate_persona_insights import PersonaInsightGenerator
            insight_generator = PersonaInsightGenerator()
            insights = insight_generator.run()
        else:
            insights = None
            
        # Generate creative credit card designs
        print("\nGenerating creative credit card designs...")
        from curate_credit_cards import CreditCardCurator
        curator = CreditCardCurator()
        card_designs = curator.run()
        
        print(f"\nPipeline completed successfully!")
        print(f"- Generated {len(cluster_insights)} customer personas")
        print(f"- Created recommendations for {len(recommendations)} customers")
        print(f"- Generated {num_emails} personalized ad emails")
        if insights:
            print(f"- Generated detailed insights for {len(insights)} personas")
        if card_designs:
            print(f"- Created {len(card_designs)} creative credit card designs")
        
        return cluster_insights, predictions, recommendations, insights, card_designs

if __name__ == "__main__":
    creator = CreateCustomerPersonas()
    cluster_insights, predictions, recommendations, insights, card_designs = creator.run(generate_insights=True)
    
    # Print example predictions
    print("\nExample Customer Predictions:")
    print("=" * 50)
    for customer in predictions[:3]:
        print(f"\nCustomer ID: {customer['customer_ID']}")
        print("Top Personas:")
        for persona in customer['persona_predictions'][:2]:
            print(f"- {persona['persona_name']}: {persona['confidence_score']:.2%} confidence")
        print("-" * 50)
    
    # Print example insights if available
    if insights:
        print("\nExample Persona Insights:")
        print("=" * 50)
        example_persona = list(insights.keys())[0]
        print(f"\nPersona: {example_persona}")
        print(insights[example_persona][:500] + "...")
    
    # Print example card design
    if card_designs:
        print("\nExample Creative Card Design:")
        print("=" * 50)
        example_persona = list(card_designs.keys())[0]
        design = card_designs[example_persona]
        print(f"\nPersona: {example_persona}")
        print(f"Card Name: {design['card_name']}")
        print(f"Design Theme: {design['design_theme']}")
        print(f"Annual Fee: ${design['annual_fee']}")
        print(f"Rewards Rate: {design['rewards_rate']*100}%")
        print(f"Welcome Bonus: {design['welcome_bonus']}")
        print("\nKey Features:")
        for feature in design['features'][:5]:
            print(f"- {feature}")
        print("\nUnique Selling Points:")
        for point in design['unique_selling_points']:
            print(f"- {point}") 