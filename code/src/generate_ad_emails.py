import json
import os
import pandas as pd
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

class AdEmailGenerator:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize OpenAI client
        self.client = OpenAI()  # Will use OPENAI_API_KEY from .env file
        
        # Load recommendations and customer data
        self.recommendations = self._create_sample_recommendations()
        self.customer_data = self._create_sample_customer_data()
    
    def _create_sample_customer_data(self):
        """Create sample customer data for demonstration"""
        data = {
            'customer_ID': ['CUST2025UE', 'CUST2025LE', 'CUST2025QB', 'CUST2025RS', 'CUST2025MK'],
            'occupation': ['Software Engineer', 'Business Analyst', 'Real Estate Agent', 'Marketing Manager', 'Healthcare Professional'],
            'interests': [
                ['Technology', 'Travel', 'Fitness'],
                ['Finance', 'Food', 'Travel'],
                ['Real Estate', 'Luxury', 'Travel'],
                ['Digital Marketing', 'Social Media', 'Travel'],
                ['Healthcare', 'Wellness', 'Travel']
            ],
            'age': [28, 35, 42, 31, 38],
            'income': [95000, 120000, 180000, 110000, 140000]
        }
        return pd.DataFrame(data)
    
    def _create_sample_recommendations(self):
        """Create sample recommendations for demonstration"""
        return [
            {
                'customer_ID': 'CUST2025UE',
                'top_personas': [
                    {'name': 'Young Professionals', 'confidence': 42.38},
                    {'name': 'Active Middle-Aged Professionals', 'confidence': 38.66}
                ],
                'recommended_cards': [
                    {
                        'card_name': 'Wells Fargo Active Cash® Card',
                        'card_details': {
                            'features': [
                                'Unlimited 2% cash rewards on purchases',
                                '$200 cash rewards bonus after spending $1,000 in first 3 months',
                                'Cell phone protection up to $600'
                            ]
                        }
                    },
                    {
                        'card_name': 'Wells Fargo Autograph℠ Card',
                        'card_details': {
                            'features': [
                                'Earn 3X points on restaurants, travel, gas stations, transit, and streaming services',
                                'Earn 1X points on other purchases',
                                '$300 cash rewards bonus after spending $3,000 in first 3 months'
                            ]
                        }
                    }
                ]
            },
            {
                'customer_ID': 'CUST2025LE',
                'top_personas': [
                    {'name': 'Established Professionals', 'confidence': 39.37},
                    {'name': 'High-Income Middle-Aged Professionals', 'confidence': 16.20}
                ],
                'recommended_cards': [
                    {
                        'card_name': 'Wells Fargo Autograph℠ Card',
                        'card_details': {
                            'features': [
                                'Earn 3X points on restaurants, travel, gas stations, transit, and streaming services',
                                'Earn 1X points on other purchases',
                                '$300 cash rewards bonus after spending $3,000 in first 3 months'
                            ]
                        }
                    },
                    {
                        'card_name': 'Wells Fargo Active Cash® Card',
                        'card_details': {
                            'features': [
                                'Unlimited 2% cash rewards on purchases',
                                '$200 cash rewards bonus after spending $1,000 in first 3 months',
                                'Cell phone protection up to $600'
                            ]
                        }
                    }
                ]
            },
            {
                'customer_ID': 'CUST2025QB',
                'top_personas': [
                    {'name': 'High-Value Real Estate Investors', 'confidence': 45.12},
                    {'name': 'Active Middle-Aged Professionals', 'confidence': 28.33}
                ],
                'recommended_cards': [
                    {
                        'card_name': 'Wells Fargo Reflect® Card',
                        'card_details': {
                            'features': [
                                'Up to 21 months 0% intro APR on purchases and qualifying balance transfers',
                                'Cell phone protection up to $600',
                                'My Wells Fargo Deals rewards'
                            ]
                        }
                    },
                    {
                        'card_name': 'Wells Fargo Active Cash® Card',
                        'card_details': {
                            'features': [
                                'Unlimited 2% cash rewards on purchases',
                                '$200 cash rewards bonus after spending $1,000 in first 3 months',
                                'Cell phone protection up to $600'
                            ]
                        }
                    }
                ]
            },
            {
                'customer_ID': 'CUST2025RS',
                'top_personas': [
                    {'name': 'Young Professionals', 'confidence': 38.45},
                    {'name': 'High-Income Middle-Aged Professionals', 'confidence': 32.18}
                ],
                'recommended_cards': [
                    {
                        'card_name': 'Wells Fargo Autograph℠ Card',
                        'card_details': {
                            'features': [
                                'Earn 3X points on restaurants, travel, gas stations, transit, and streaming services',
                                'Earn 1X points on other purchases',
                                '$300 cash rewards bonus after spending $3,000 in first 3 months'
                            ]
                        }
                    },
                    {
                        'card_name': 'Wells Fargo Active Cash® Card',
                        'card_details': {
                            'features': [
                                'Unlimited 2% cash rewards on purchases',
                                '$200 cash rewards bonus after spending $1,000 in first 3 months',
                                'Cell phone protection up to $600'
                            ]
                        }
                    }
                ]
            },
            {
                'customer_ID': 'CUST2025MK',
                'top_personas': [
                    {'name': 'Established Professionals', 'confidence': 41.23},
                    {'name': 'Active Middle-Aged Professionals', 'confidence': 35.67}
                ],
                'recommended_cards': [
                    {
                        'card_name': 'Wells Fargo Active Cash® Card',
                        'card_details': {
                            'features': [
                                'Unlimited 2% cash rewards on purchases',
                                '$200 cash rewards bonus after spending $1,000 in first 3 months',
                                'Cell phone protection up to $600'
                            ]
                        }
                    },
                    {
                        'card_name': 'Wells Fargo Autograph℠ Card',
                        'card_details': {
                            'features': [
                                'Earn 3X points on restaurants, travel, gas stations, transit, and streaming services',
                                'Earn 1X points on other purchases',
                                '$300 cash rewards bonus after spending $3,000 in first 3 months'
                            ]
                        }
                    }
                ]
            }
        ]
    
    def _generate_email_content(self, customer_info, recommendations):
        """Generate personalized email content using OpenAI"""
        # Extract customer details
        customer_id = customer_info['customer_ID']
        top_personas = recommendations['top_personas']
        recommended_cards = recommendations['recommended_cards']
        
        # Create a detailed prompt for GPT
        prompt = f"""Generate a personalized credit card recommendation email with the following details:

Customer Profile:
- Occupation: {customer_info['occupation']}
- Age: {customer_info['age']}
- Annual Income: ${customer_info['income']:,}
- Interests: {', '.join(customer_info['interests'])}
- Primary Persona: {top_personas[0]['name']} ({top_personas[0]['confidence']:.1f}% confidence)
- Secondary Persona: {top_personas[1]['name']} ({top_personas[1]['confidence']:.1f}% confidence)

Primary Card Recommendation:
{recommended_cards[0]['card_name']}
Features:
{chr(10).join('- ' + feature for feature in recommended_cards[0]['card_details']['features'])}

Secondary Card Recommendation:
{recommended_cards[1]['card_name']}
Features:
{chr(10).join('- ' + feature for feature in recommended_cards[1]['card_details']['features'])}

Requirements:
1. Write a professional yet warm email
2. Include a compelling subject line
3. Personalize the content based on the customer's profile and interests
4. Highlight how each card's features align with their lifestyle
5. Include a clear call to action
6. Keep it concise (max 300 words)
7. Format with "Subject: [subject]" on first line, followed by two newlines and then the email content
"""

        try:
            # Generate email content using GPT-4 Mini
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert credit card marketing specialist who writes highly personalized and compelling email recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            # Extract the generated email
            email_content = response.choices[0].message.content.strip()
            return email_content
            
        except Exception as e:
            print(f"Error generating email for customer {customer_id}: {str(e)}")
            return None
    
    def generate_emails(self, output_dir='../output/ad_emails'):
        """Generate personalized emails for all customers"""
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found in .env file.")
            print("Please add your OpenAI API key to the .env file:")
            print("OPENAI_API_KEY=your-api-key-here")
            return 0
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate emails for each customer
        emails_generated = 0
        for recommendation in self.recommendations:
            customer_id = recommendation['customer_ID']
            
            # Get customer info from dataframe
            customer_info = self.customer_data[self.customer_data['customer_ID'] == customer_id].iloc[0].to_dict()
            
            print(f"\nGenerating email for customer {customer_id}...")
            
            # Generate email content
            email_content = self._generate_email_content(customer_info, recommendation)
            
            if email_content:
                # Save email to file
                filename = f"{customer_id}_email_{datetime.now().strftime('%Y%m%d')}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(email_content)
                
                emails_generated += 1
                print(f"Email generated and saved to {filepath}")
        
        print(f"\nCompleted! Generated {emails_generated} personalized email advertisements.")
        return emails_generated

if __name__ == "__main__":
    print("Note: This script requires an OpenAI API key in the .env file.")
    print("Please ensure you have added your API key to the .env file before running.\n")
    
    # Generate personalized ad emails
    generator = AdEmailGenerator()
    num_emails = generator.generate_emails()
    
    if num_emails > 0:
        # Print example email
        example_file = os.listdir('../output/ad_emails')[0]
        print("\nExample Generated Email:")
        print("=" * 50)
        with open(f'../output/ad_emails/{example_file}', 'r', encoding='utf-8') as f:
            print(f.read())