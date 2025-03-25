import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class CreditCardRecommender:
    def __init__(self):
        """Initialize the credit card recommender with Wells Fargo credit cards"""
        self.credit_cards = {
            "Wells Fargo Active Cash® Card": {
                "annual_fee": 0,
                "rewards_rate": 0.02,
                "welcome_bonus": "200",
                "credit_score_required": 670,
                "income_requirement": 50000,
                "best_for": ["Established Professionals"],
                "features": [
                    "2% cash rewards on all purchases",
                    "$200 cash rewards bonus",
                    "Cell phone protection",
                    "Zero liability protection",
                    "No annual fee",
                    "No category restrictions"
                ],
                "category_rewards": {
                    "travel": 0.02,
                    "dining": 0.02,
                    "gas": 0.02,
                    "streaming": 0.02,
                    "other": 0.02
                },
                "card_type": "cash_back",
                "primary_benefit": "flat_rate_rewards"
            },
            "Wells Fargo Autograph℠ Card": {
                "annual_fee": 0,
                "rewards_rate": 0.03,
                "welcome_bonus": "20000",
                "credit_score_required": 670,
                "income_requirement": 50000,
                "best_for": ["Young Professionals", "Active Middle-Aged Professionals"],
                "features": [
                    "3x points on restaurants and takeout",
                    "3x points on gas and transit",
                    "3x points on streaming services",
                    "2x points on travel",
                    "20,000 bonus points welcome offer",
                    "Cell phone protection",
                    "No foreign transaction fees",
                    "No annual fee"
                ],
                "category_rewards": {
                    "travel": 0.02,
                    "dining": 0.03,
                    "gas": 0.03,
                    "streaming": 0.03,
                    "other": 0.01
                },
                "card_type": "travel_rewards",
                "primary_benefit": "everyday_rewards"
            },
            "Wells Fargo Reflect® Card": {
                "annual_fee": 0,
                "rewards_rate": 0.01,
                "welcome_bonus": "0",
                "credit_score_required": 670,
                "income_requirement": 50000,
                "best_for": ["High-Income Middle-Aged Professionals"],
                "features": [
                    "Up to 21 months 0% APR on purchases",
                    "Up to 21 months 0% APR on balance transfers",
                    "Cell phone protection",
                    "Roadside dispatch",
                    "Zero liability protection",
                    "No annual fee"
                ],
                "category_rewards": {
                    "travel": 0.01,
                    "dining": 0.01,
                    "gas": 0.01,
                    "streaming": 0.01,
                    "other": 0.01
                },
                "card_type": "low_interest",
                "primary_benefit": "zero_interest"
            },
            "Wells Fargo Propel American Express® card": {
                "annual_fee": 0,
                "rewards_rate": 0.035,
                "welcome_bonus": "20000",
                "credit_score_required": 670,
                "income_requirement": 50000,
                "best_for": ["High-Value Real Estate Investors"],
                "features": [
                    "3.5x points on flights and hotels",
                    "3x points on car rentals",
                    "2x points on dining and takeout",
                    "2x points on streaming services",
                    "20,000 bonus points welcome offer",
                    "American Express benefits",
                    "Travel accident insurance",
                    "Lost luggage reimbursement",
                    "Return protection",
                    "Extended warranty",
                    "No annual fee"
                ],
                "category_rewards": {
                    "travel": 0.035,
                    "dining": 0.02,
                    "gas": 0.02,
                    "streaming": 0.02,
                    "other": 0.01
                },
                "card_type": "travel_rewards",
                "primary_benefit": "premium_travel"
            }
        }
        
        # Define persona preferences for credit cards with spending patterns
        self.persona_preferences = {
            "High-Value Real Estate Investors": {
                "preferred_features": ["travel_benefits", "purchase_protection", "premium_rewards"],
                "min_rewards_rate": 0.02,
                "max_annual_fee": 95,
                "spending_patterns": {
                    "travel": 0.35,
                    "dining": 0.20,
                    "gas": 0.10,
                    "streaming": 0.05,
                    "other": 0.30
                },
                "preferred_card_types": ["travel_rewards"],
                "preferred_benefits": ["premium_travel"]
            },
            "Established Professionals": {
                "preferred_features": ["cash_back", "purchase_protection", "no_category_restrictions"],
                "min_rewards_rate": 0.015,
                "max_annual_fee": 0,
                "spending_patterns": {
                    "travel": 0.15,
                    "dining": 0.25,
                    "gas": 0.15,
                    "streaming": 0.10,
                    "other": 0.35
                },
                "preferred_card_types": ["cash_back"],
                "preferred_benefits": ["flat_rate_rewards"]
            },
            "Young Professionals": {
                "preferred_features": ["travel_rewards", "dining_rewards", "streaming_rewards"],
                "min_rewards_rate": 0.02,
                "max_annual_fee": 0,
                "spending_patterns": {
                    "travel": 0.20,
                    "dining": 0.30,
                    "gas": 0.10,
                    "streaming": 0.15,
                    "other": 0.25
                },
                "preferred_card_types": ["travel_rewards"],
                "preferred_benefits": ["everyday_rewards"]
            },
            "High-Income Middle-Aged Professionals": {
                "preferred_features": ["balance_transfer", "purchase_protection", "zero_apr"],
                "min_rewards_rate": 0.01,
                "max_annual_fee": 0,
                "spending_patterns": {
                    "travel": 0.25,
                    "dining": 0.20,
                    "gas": 0.15,
                    "streaming": 0.05,
                    "other": 0.35
                },
                "preferred_card_types": ["low_interest"],
                "preferred_benefits": ["zero_interest"]
            },
            "Active Middle-Aged Professionals": {
                "preferred_features": ["travel_rewards", "dining_rewards", "gas_rewards"],
                "min_rewards_rate": 0.02,
                "max_annual_fee": 0,
                "spending_patterns": {
                    "travel": 0.25,
                    "dining": 0.25,
                    "gas": 0.15,
                    "streaming": 0.10,
                    "other": 0.25
                },
                "preferred_card_types": ["travel_rewards"],
                "preferred_benefits": ["everyday_rewards"]
            }
        }
    
    def recommend_cards(self, customer_predictions):
        """Recommend credit cards based on customer persona predictions"""
        recommendations = []
        
        # Define primary card assignments for each persona
        primary_card_assignments = {
            "High-Value Real Estate Investors": "Wells Fargo Propel American Express® card",
            "Established Professionals": "Wells Fargo Active Cash® Card",
            "Young Professionals": "Wells Fargo Autograph℠ Card",
            "High-Income Middle-Aged Professionals": "Wells Fargo Reflect® Card",
            "Active Middle-Aged Professionals": "Wells Fargo Autograph℠ Card"
        }
        
        # Define secondary card preferences
        secondary_card_preferences = {
            "High-Value Real Estate Investors": ["Wells Fargo Autograph℠ Card", "Wells Fargo Active Cash® Card"],
            "Established Professionals": ["Wells Fargo Autograph℠ Card", "Wells Fargo Propel American Express® card"],
            "Young Professionals": ["Wells Fargo Propel American Express® card", "Wells Fargo Active Cash® Card"],
            "High-Income Middle-Aged Professionals": ["Wells Fargo Active Cash® Card", "Wells Fargo Autograph℠ Card"],
            "Active Middle-Aged Professionals": ["Wells Fargo Propel American Express® card", "Wells Fargo Active Cash® Card"]
        }
        
        for customer in customer_predictions:
            # Get top 2 personas with their confidence scores
            top_personas = customer['persona_predictions'][:2]
            primary_persona = top_personas[0]['persona_name']
            secondary_persona = top_personas[1]['persona_name']
            
            # Calculate card scores
            card_scores = {card: 0 for card in self.credit_cards}
            
            # Assign primary card based on dominant persona
            primary_card = primary_card_assignments[primary_persona]
            card_scores[primary_card] = top_personas[0]['confidence_score'] * 5.0
            
            # Calculate scores for secondary cards
            for persona in top_personas:
                persona_name = persona['persona_name']
                confidence = persona['confidence_score']
                preferences = self.persona_preferences[persona_name]
                
                # Add spending pattern score
                for card_name, card_info in self.credit_cards.items():
                    if card_name != primary_card:  # Skip primary card
                        weighted_rewards = sum(
                            preferences['spending_patterns'][category] * card_info['category_rewards'][category]
                            for category in preferences['spending_patterns']
                        )
                        card_scores[card_name] += weighted_rewards * confidence * 2.0
                        
                        # Boost score for preferred secondary cards
                        if card_name in secondary_card_preferences[persona_name]:
                            card_scores[card_name] += confidence * 2.0
                        
                        # Add feature match score
                        feature_matches = sum(
                            1 for feature in preferences['preferred_features']
                            if any(feature in f.lower() for f in card_info['features'])
                        )
                        card_scores[card_name] += (feature_matches / len(preferences['preferred_features'])) * confidence
            
            # Ensure primary card stays on top
            max_secondary_score = max(score for card, score in card_scores.items() if card != primary_card)
            card_scores[primary_card] = max_secondary_score * 1.2
            
            # Get final recommendations
            recommended_cards = sorted(card_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Create recommendation object
            recommendation = {
                'customer_ID': customer['customer_ID'],
                'user_features': customer['user_features'],
                'top_personas': [
                    {
                        'name': p['persona_name'],
                        'confidence': p['confidence_score']
                    } for p in top_personas
                ],
                'recommended_cards': [
                    {
                        'card_name': card_name,
                        'score': float(score),
                        'card_details': self.credit_cards[card_name],
                        'match_reasons': self._get_match_reasons(card_name, top_personas)
                    } for card_name, score in recommended_cards[:2]
                ]
            }
            recommendations.append(recommendation)
        
        # Save recommendations
        self._save_recommendations(recommendations)
        
        return recommendations
    
    def _get_match_reasons(self, card_name, top_personas):
        """Generate personalized reasons why the card matches the customer's personas"""
        reasons = []
        card_info = self.credit_cards[card_name]
        
        for persona in top_personas:
            persona_name = persona['persona_name']
            preferences = self.persona_preferences[persona_name]
            
            # Check primary benefit match
            if card_info['primary_benefit'] in preferences['preferred_benefits']:
                benefit_name = card_info['primary_benefit'].replace('_', ' ').title()
                reasons.append(f"Matches {persona_name} preference for {benefit_name}")
            
            # Calculate category rewards
            category_rewards = []
            for category, spending in preferences['spending_patterns'].items():
                reward_rate = card_info['category_rewards'][category]
                if reward_rate > 0.01:  # Only mention categories with above-average rewards
                    category_rewards.append(f"{reward_rate*100}% back on {category}")
            
            if category_rewards:
                reasons.append(f"Matches {persona_name} spending with {', '.join(category_rewards)}")
            
            # Check feature matches
            feature_matches = []
            for feature in preferences['preferred_features']:
                if any(feature in f.lower() for f in card_info['features']):
                    feature_matches.append(feature.replace('_', ' '))
            
            if feature_matches:
                reasons.append(f"Offers {persona_name} preferred features: {', '.join(feature_matches)}")
        
        return reasons
    
    def _save_recommendations(self, recommendations):
        """Save recommendations to JSON and CSV files"""
        # Create artifacts directory if it doesn't exist
        artifacts_dir = '../artifacts'
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save detailed recommendations to JSON
        output_json = os.path.join(artifacts_dir, 'credit_card_recommendations.json')
        with open(output_json, 'w') as f:
            json.dump(recommendations, f, indent=4)
        
        # Create DataFrame for CSV with top 2 cards for each customer
        csv_data = []
        for rec in recommendations:
            top_cards = rec['recommended_cards'][:2]
            row = {
                'customer_ID': rec['customer_ID'],
                'primary_persona': rec['top_personas'][0]['name'],
                'primary_persona_confidence': rec['top_personas'][0]['confidence'],
                'secondary_persona': rec['top_personas'][1]['name'],
                'secondary_persona_confidence': rec['top_personas'][1]['confidence'],
                'recommended_card_1': top_cards[0]['card_name'],
                'card_1_score': top_cards[0]['score'],
                'card_1_match_reasons': '; '.join(top_cards[0]['match_reasons']),
                'recommended_card_2': top_cards[1]['card_name'],
                'card_2_score': top_cards[1]['score'],
                'card_2_match_reasons': '; '.join(top_cards[1]['match_reasons']),
                **rec['user_features']
            }
            csv_data.append(row)
        
        # Save recommendations to CSV
        output_csv = os.path.join(artifacts_dir, 'credit_card_recommendations.csv')
        pd.DataFrame(csv_data).to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Load persona predictions
    with open('../artifacts/persona_predictions.json', 'r') as f:
        predictions = json.load(f)
    
    # Create recommender and get recommendations
    recommender = CreditCardRecommender()
    recommendations = recommender.recommend_cards(predictions)
    
    # Calculate card recommendation distribution
    total_customers = len(recommendations)
    card_distribution = {
        "Primary Recommendations": {},
        "Secondary Recommendations": {}
    }
    
    for rec in recommendations:
        # Count primary recommendations
        primary_card = rec['recommended_cards'][0]['card_name']
        card_distribution["Primary Recommendations"][primary_card] = card_distribution["Primary Recommendations"].get(primary_card, 0) + 1
        
        # Count secondary recommendations
        secondary_card = rec['recommended_cards'][1]['card_name']
        card_distribution["Secondary Recommendations"][secondary_card] = card_distribution["Secondary Recommendations"].get(secondary_card, 0) + 1
    
    # Print distribution statistics
    print("\nCredit Card Recommendation Distribution:")
    print("=" * 50)
    print(f"Total Customers: {total_customers}")
    print("\nPrimary Recommendations:")
    for card, count in card_distribution["Primary Recommendations"].items():
        percentage = (count / total_customers) * 100
        print(f"- {card}: {count} customers ({percentage:.1f}%)")
    
    print("\nSecondary Recommendations:")
    for card, count in card_distribution["Secondary Recommendations"].items():
        percentage = (count / total_customers) * 100
        print(f"- {card}: {count} customers ({percentage:.1f}%)")
    
    # Print example recommendations for different persona combinations
    print("\nExample Credit Card Recommendations:")
    print("=" * 50)
    
    # Find customers with different persona combinations
    persona_combinations = {}
    for rec in recommendations:
        combo = tuple(sorted([p['name'] for p in rec['top_personas']]))
        if combo not in persona_combinations:
            persona_combinations[combo] = rec
    
    # Show recommendations for different combinations
    for combo, rec in list(persona_combinations.items())[:3]:
        print(f"\nCustomer ID: {rec['customer_ID']}")
        print("Top Personas:")
        for p in rec['top_personas']:
            print(f"- {p['name']}: {p['confidence']:.2%} confidence")
        print("\nRecommended Cards:")
        for card in rec['recommended_cards']:
            print(f"- {card['card_name']}: Score {card['score']:.2f}")
            print(f"  Match Reasons: {', '.join(card['match_reasons'])}")
            print(f"  Features: {', '.join(card['card_details']['features'])}") 