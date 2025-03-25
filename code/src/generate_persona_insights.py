import json
import os
from openai import OpenAI
from dotenv import load_dotenv

class PersonaInsightGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.client = OpenAI()
    
    def load_cluster_insights(self):
        """Load cluster insights from the artifacts directory"""
        with open('../artifacts/cluster_insights.json', 'r') as f:
            return json.load(f)
    
    def generate_insights(self, cluster_data):
        """Generate detailed insights for a single persona/cluster"""
        prompt = f"""Analyze this customer persona and provide detailed business insights:

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

Please provide insights in the following areas:
1. Financial Behavior Analysis
2. Marketing Strategy Recommendations
3. Product Development Opportunities
4. Risk Assessment
5. Customer Retention Strategies
6. Cross-Selling Opportunities
7. Digital Engagement Recommendations
8. Competitive Analysis

Format the response as a structured report with clear sections and bullet points."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing insights about customer personas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating insights for {cluster_data['name']}: {str(e)}")
            return None
    
    def save_insights(self, all_insights):
        """Save generated insights to a file"""
        output_dir = '../output/persona_insights'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual persona insights
        for persona_name, insights in all_insights.items():
            filename = f"{persona_name.lower().replace(' ', '_')}_insights.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(insights)
        
        # Save combined insights report
        report_path = os.path.join(output_dir, 'complete_persona_insights.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CUSTOMER PERSONA INSIGHTS REPORT\n")
            f.write("==============================\n\n")
            for persona_name, insights in all_insights.items():
                f.write(f"\n{persona_name}\n")
                f.write("=" * len(persona_name) + "\n\n")
                f.write(insights)
                f.write("\n\n" + "-"*80 + "\n\n")
    
    def run(self):
        """Run the complete insight generation process"""
        print("\nGenerating AI-powered persona insights...")
        
        # Load cluster insights
        cluster_insights = self.load_cluster_insights()
        
        # Generate insights for each persona
        all_insights = {}
        for cluster in cluster_insights:
            print(f"\nAnalyzing {cluster['name']}...")
            insights = self.generate_insights(cluster)
            if insights:
                all_insights[cluster['name']] = insights
        
        # Save all insights
        self.save_insights(all_insights)
        
        print(f"\nInsights generation completed!")
        print(f"- Generated insights for {len(all_insights)} personas")
        print(f"- Saved to ../output/persona_insights/")
        
        return all_insights

if __name__ == "__main__":
    generator = PersonaInsightGenerator()
    insights = generator.run()
    
    # Print example insight
    if insights:
        print("\nExample Insight:")
        print("=" * 50)
        example_persona = list(insights.keys())[0]
        print(f"\n{example_persona}:")
        print(insights[example_persona][:500] + "...") 