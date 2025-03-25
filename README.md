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

## 🎯 Introduction
This project uses machine learning and AI to implement an advanced customer persona analysis and credit card recommendation system. It processes customer data to create detailed personas, generates personalized credit card recommendations, and designs tailored credit-card products for each customer segment. This project is for the BANK to optimize its credit card sales offerings. 

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
Pansandida Aurat.

## ⚙️ What It Does
Our project takes in the data of the users, which are customer_transaction_history.csv, customer_profiles.csv, and customer_social_sentiments.csv (which are sourced using open-source AI models by intelligent prompt-engineering) and creates buckets of similar customers, which we are calling "Customer Personas" (CP for short) and gives the bank the information about these CPs. Then, we predict which two CPs each customer is a part of using ML modeling, based on which we recommend two of the most suited credit cards that Wells Fargo already offers. Then we go a step beyond in curating the perfect email for each customer using data that we have to entice them about their AI-matched Credit Card(s). And we don't just stop there; we help Wells Fargo CURATE new credit cards, which will perform better in the market (high adoption rate) using AI on the CP buckets.

## 🛠️ How We Built It
Core Technologies:
Python (3.12+) - Main programming language
scikit-learn - For machine learning and clustering
pandas - For data processing and analysis
numpy - For numerical computations
matplotlib/seaborn - For data visualization
AI/ML Components:
OpenAI API - For generating card designs and insights
K-means clustering - For customer segmentation
Sentiment analysis - For social media data processing
Data Storage:
JSON/CSV - For structured data storage
File-based storage system for outputs
Development Tools:
Git - Version control
Python virtual environment
Environment variables for API key management.

## 🚧 Challenges We Faced
The main challenge was to find time outside of doing our JIRAS. Be that as it may, this was a busy sprint for the three of us, and the hackathon is at the sprint end, so it got more difficult to get the time for this. The other challenge was data. We didn't want to fabricate the data as we were trying to learn the patterns in the data, and if the data itself is random, then your models can't do anything. But then we intelligently prompted the open-source AIs for the data and curated some 10k transaction rows for 676 customers. There were no other challenges, only learnings which were quite fun for us.

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone [https://github.com/your-repo.git](https://github.com/ewfx/aidhp-dating-dataset.git)
   ```
2. Install dependencies  
   ```sh
   pip install -r requirements.txt (for Python)
   ```
3. Run the project  
   ```sh
   python create_customer_personas.py
   ```

## 🏗️ Tech Stack
- 🔹 Frontend: No time to build it (JIRAS smh)
- 🔹 Backend: Python, pandas, scikit, numpy
- 🔹 Database: Not required as small dataset
- 🔹 Other: OpenAI API, K means clustering, Sentiment Analysis

## 👥 Team
- **Tanish Singhal** - [GitHub](#) | [LinkedIn](#)
- **Harshita Taparia** - [GitHub](#) | [LinkedIn](#)
- **Shankha Nayek** - [GitHub](#) | [LinkedIn](#)
