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
A brief overview of your project and its purpose. Mention which problem statement are your attempting to solve. Keep it concise and engaging.

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
What inspired you to create this project? Describe the problem you're solving.

## ⚙️ What It Does
Explain the key features and functionalities of your project.

## 🛠️ How We Built It
Briefly outline the technologies, frameworks, and tools used in development.

## 🚧 Challenges We Faced
Describe the major technical or non-technical challenges your team encountered.

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/your-repo.git
   ```
2. Install dependencies  
   ```sh
   npm install  # or pip install -r requirements.txt (for Python)
   ```
3. Run the project  
   ```sh
   npm start  # or python app.py
   ```

## 🏗️ Tech Stack
- 🔹 Frontend: React / Vue / Angular
- 🔹 Backend: Node.js / FastAPI / Django
- 🔹 Database: PostgreSQL / Firebase
- 🔹 Other: OpenAI API / Twilio / Stripe

## 👥 Team
- **Your Name** - [GitHub](#) | [LinkedIn](#)
- **Teammate 2** - [GitHub](#) | [LinkedIn](#)

# Customer Persona Analysis

This project analyzes customer data to create meaningful customer personas using machine learning clustering techniques. It processes customer profiles, transaction history, and social sentiment data to identify distinct customer segments.

## Data Requirements

The code expects three CSV files in the `code/resources` directory:

1. `customer_profiles.csv`:
   - customer_ID
   - Age
   - Interests (comma-separated)
   - Income per Year (in dollars)
   - Occupation

2. `customer_transaction_history.csv`:
   - customer_ID
   - product_ID
   - Transaction Type
   - Category
   - Amt (in dollars)
   - Payment Date
   - Payment Mode

3. `customer_social_sentiments.csv`:
   - customer_ID
   - Platform
   - Content
   - TimeStamp
   - Sentiment Score
   - Intent

## Setup and Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python code/src/create_customer_personas.py
```

The script will:
1. Load and preprocess the data
2. Create customer clusters using K-means clustering
3. Generate insights for each cluster
4. Save cluster labels to `code/artifacts/cluster_labels.csv`
5. Create visualizations in `code/artifacts/cluster_visualization.png`

## Output

The script provides:
1. Detailed statistics for each cluster including:
   - Cluster size
   - Average age and income
   - Transaction patterns
   - Sentiment analysis
   - Common interests and occupations

2. Visualizations showing:
   - Age vs Income Distribution
   - Transaction Amount vs Frequency
   - Sentiment vs Engagement
   - Total Spend Distribution by Cluster

## Directory Structure

```
.
├── code/
│   ├── src/
│   │   └── create_customer_personas.py
│   ├── resources/
│   │   ├── customer_profiles.csv
│   │   ├── customer_transaction_history.csv
│   │   └── customer_social_sentiments.csv
│   └── artifacts/
│       ├── cluster_labels.csv
│       └── cluster_visualization.png
├── requirements.txt
└── README.md
```