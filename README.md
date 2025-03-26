# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## Introduction
This project uses machine learning and AI to implement an advanced customer persona analysis and credit card recommendation system. It processes customer data to create detailed personas, generates personalized credit card recommendations, and designs tailored products for each customer segment. This project is for the BANK to optimize its credit card sales offerings. Note: We have limited our offerings to credit cards only; please review the Report document for more details.

## Demo PLEASE READ THE dating_dataset_REPORT.pdf
🔗 [Live Demo](#) (not applicable)  
📹 [Video Demo](#) (not applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## Inspiration
We are all credit card enthusiasts and hold multiple cards each. However, we do not want these many credit cards as they are difficult to manage. There should be a better way to handle customer credit card distribution, and then this Hackathon came. We wanted to suggest the best available credit cards to the customers and CURATE new, exciting, and more insight-driven cards. Since customer segmentation is a big problem and we see sub-clusters inside clusters as well, it's very difficult for humans to create credit cards to cover the whole customer segment, which inevitably results in over-simplification of the customer personas, pushing the same cards to different segments thus leading to low adoption rate. What we wanted to do and done here is to curate credit cards using data-driven methods so that the Customer Adoption Rate is high and everybody gets their dream credit cards.

## What It Does
Our project takes in the data of the users, which are customer_transaction_history.csv, customer_profiles.csv, and customer_social_sentiments.csv (which are sourced using open-source AI models by intelligent prompt-engineering) and creates buckets of similar customers, which we are calling "Customer Personas" (CP for short) and gives the bank the information about these CPs. Then, we predict which two CPs each customer is a part of using ML modeling, based on which we recommend two of the most suited credit cards that Wells Fargo already offers. Then we go a step beyond in curating the perfect email for each customer using data that we have to entice them about their AI-matched Credit Card(s). And we don't just stop there; we help Wells Fargo CURATE new credit cards, which will perform better in the market (high adoption rate) using AI on the CP buckets. If we adopt this AI (and improve it further), we can be the leaders in the Credit Card Segment of the banking industry.

### Processes Customer Transaction Data
Analyzes spending categories and payment patterns.

### Profiles Each Customer
Uses age, occupation, and stated interests from customer_profiles.csv.

### Incorporates Social Sentiments
Extracts additional behavioral cues from social media posts.

### Builds a Customer Persona
Merges all sources to define unique tags like ["travel", "food", "tech"].

### Recommends Credit Cards
Matches personas to a predefined credit card catalog using cosine similarity.

### Generates Custom Cards (GPT-3.5)
When no good match is found (score < 0.6), GPT curates a brand new card with name and features.

### Outputs Recommendation File
A final CSV with customer ID, persona tags, recommended or curated card, and match score.

## How We Built It

### Data Merging & Cleaning
Combined all 3 datasets on customer_ID.

### Tag Extraction
mapped transaction categories + profile interests into tags.

### Vectorization
Used MultiLabelBinarizer to one-hot encode tags for cosine similarity.

### Card Matching
Compared each customer vector to a set of predefined credit card vectors.

### GPT Integration
Called OpenAI API with user context to generate card details when needed.

### Export
Saved results as customer_credit_card_recommendations.csv.

## Challenges We Faced
1. The main challenge was data. We didn't want to fabricate the data as we were trying to learn the patterns in the data, and if the data itself is random, then your models can't do anything. But then we intelligently prompted the open-source AIs for the data and curated some 10k transaction rows for 676 customers.
2. The second is the time limitation: there is so much we can do here, so much we wanted to do here, but the time constraints of the challenge and team members' availability mid-week only allowed us to do this much.

## How to Run
1. Clone the repository  
   ```sh
   git clone [https://github.com/your-repo.git](https://github.com/ewfx/aidhp-dating-dataset.git)
   ```
2. Install dependencies  
   ```sh
   pip install -r requirements.txt
   ```
3. Run the project  
   ```sh
   python create_customer_personas.py
   ```
   There is a .env file which contains OPENAI_API_KEY in which the user can paste their API key.

## Tech Stack
- 🔹 Frontend: No time to build it (JIRAS smh)
- 🔹 Backend: Python, pandas, scikit, NumPy
- 🔹 Database: Not required as small dataset
- 🔹 Other: OpenAI API, K means clustering, Sentiment Analysis

## Team
- **Tanish Singhal** - [GitHub](#) | [LinkedIn](#)
- **Harshita Taparia** - [GitHub](#) | [LinkedIn](#)
- **Shankha Nayek** - [GitHub](#) | [LinkedIn](#)
