import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

class CreditCardCurator:
    def __init__(self):
        """Initialize the credit card curator with OpenAI client"""
        # Load environment variables
        load_dotenv()
        
        # Debug: Print the API key (first few characters)
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"API Key loaded (first 10 chars): {api_key[:10]}...")
        else:
            print("Warning: No API key found in environment variables")
        
        # Initialize OpenAI client with explicit API key
        self.client = OpenAI(api_key=api_key)
    
    def load_cluster_insights(self):
        """Load cluster insights from the artifacts directory"""
        with open('../artifacts/cluster_insights.json', 'r') as f:
            return json.load(f)
    
    def generate_card_design(self, cluster_data):
        """Generate a creative credit card design for a specific persona"""
        prompt = f"""Design a creative credit card for this customer persona:

Persona Details:
- Name: {cluster_data['name']}
- Size: {cluster_data['size']} customers
- Demographics:
  * Average Age: {cluster_data['avg_age']:.1f} years
  * Average Income: ${cluster_data['avg_income']:,.2f}
  * Average Transaction: ${cluster_data['avg_transaction']:,.2f}
  * Transaction Frequency: {cluster_data['transaction_freq']:.1f} per year
  * Sentiment Score: {cluster_data['avg_sentiment']:.2f}
- Common Interests: {', '.join(cluster_data['common_interests'])}
- Common Occupations: {', '.join(cluster_data['common_occupations'])}

Please design a credit card that would appeal to this persona. Include:
1. Card Name and Branding
2. Annual Fee
3. Rewards Structure
4. Welcome Bonus
5. Key Features and Benefits
6. Credit Score Requirements
7. Income Requirements
8. Target Spending Categories
9. Unique Selling Points
10. Design Theme and Visual Elements

Format the response as a structured JSON object with these fields:
{{
    "card_name": "string",
    "annual_fee": number,
    "rewards_rate": number,
    "welcome_bonus": "string",
    "credit_score_required": number,
    "income_requirement": number,
    "best_for": ["string"],
    "features": ["string"],
    "category_rewards": {{
        "travel": number,
        "dining": number,
        "gas": number,
        "streaming": number,
        "other": number
    }},
    "card_type": "string",
    "primary_benefit": "string",
    "design_theme": "string",
    "visual_elements": ["string"],
    "unique_selling_points": ["string"]
}}"""

        try:
            print(f"\nSending request for {cluster_data['name']}...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a credit card product designer specializing in creating innovative and targeted credit card products. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Debug: Print the raw response
            print(f"Raw response: {response.choices[0].message.content.strip()[:100]}...")
            
            # Get the response content and clean it
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json prefix
            if content.endswith('```'):
                content = content[:-3]  # Remove ``` suffix
            
            # Try to parse the cleaned response
            try:
                return json.loads(content.strip())
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print("Cleaned response content:", content.strip())
                return None
                
        except Exception as e:
            print(f"Error generating card design for {cluster_data['name']}: {str(e)}")
            return None
    
    def save_card_designs(self, all_designs):
        """Save generated card designs to files"""
        output_dir = '../output/credit_card_designs'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual card designs
        for persona_name, design in all_designs.items():
            if design:  # Only save if design was generated successfully
                filename = f"{persona_name.lower().replace(' ', '_')}_card_design.json"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(design, f, indent=4)
        
        # Save combined designs report
        report_path = os.path.join(output_dir, 'all_card_designs.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(all_designs, f, indent=4)
    
    def run(self):
        """Run the complete card design generation process"""
        print("\nGenerating creative credit card designs...")
        
        # Load cluster insights
        cluster_insights = self.load_cluster_insights()
        
        # Generate card designs for each persona
        all_designs = {}
        for cluster in cluster_insights:
            print(f"\nDesigning card for {cluster['name']}...")
            design = self.generate_card_design(cluster)
            if design:
                all_designs[cluster['name']] = design
        
        # Save all designs
        self.save_card_designs(all_designs)
        
        print(f"\nCard design generation completed!")
        print(f"- Generated designs for {len(all_designs)} personas")
        print(f"- Saved to ../output/credit_card_designs/")
        
        return all_designs

if __name__ == "__main__":
    curator = CreditCardCurator()
    designs = curator.run()
    
    # Print example design
    if designs:
        print("\nExample Card Design:")
        print("=" * 50)
        example_persona = list(designs.keys())[0]
        design = designs[example_persona]
        print(f"\nCard Name: {design['card_name']}")
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