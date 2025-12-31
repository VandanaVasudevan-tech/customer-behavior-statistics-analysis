# customer-behavior-statistics-analysis
Statistical analysis of customer behavior using Python to study purchase patterns, churn, campaign performance, and regional differences. The project applies descriptive statistics, hypothesis testing, confidence intervals, and the Central Limit Theorem to derive meaningful business insights.

📌 Project Overview

    This project analyzes customer behavior using statistical and mathematical methods in Python. The goal is to identify factors affecting customer retention,    spending patterns, and campaign performance using real-world data.


🎯 Objectives

    Handle missing values in customer datasets
    
    Analyze purchase behavior using descriptive statistics
    
    Detect outliers and study distribution properties (skewness & kurtosis)
    
    Test hypotheses on gender, campaigns, product categories, and regions
    
    Examine relationships between variables using t-tests, ANOVA, and Chi-Square tests
    
    Apply the Central Limit Theorem and calculate confidence intervals


🧰 Tools & Libraries

    Python
    
    Pandas – data manipulation
    
    NumPy – numerical computations
    
    SciPy – statistical tests (t-test, ANOVA, Chi-square)
    
    Matplotlib – data visualization


📂 Dataset

    The dataset (customer_behavior.csv) contains:
    
    CustomerID – Unique customer identifier
    
    Gender – Male/Female/Unknown
    
    Region – Customer location (Unknown if missing)
    
    PurchaseAmount – Amount spent by the customer
    
    ProductCategory – Category of product purchased
    
    CampaignGroup – Email campaign group (A/B/Unknown)
    
    Churn – Customer churn status


📈 Analysis Performed

1. Data Cleaning

        Missing categorical values filled with "Unknown"
      
        Missing numerical values (PurchaseAmount) imputed with median
      
        Rows with missing Churn removed

2. Descriptive Statistics

        Average, median, and mode of PurchaseAmount
      
        Detected 22 outliers using IQR
      
        Skewness = 0.11, Kurtosis = −0.18

3. Hypothesis Testing

        Gender vs PurchaseAmount: t-test → significant difference (p = 0.0259)
      
        Campaign A vs B: t-test → no significant difference (p = 0.1703)
      
        Region vs PurchaseAmount: ANOVA → no significant difference (p = 0.767)
      
        ProductCategory vs Churn: Chi-Square test → no significant relationship (p = 0.4415)

4. Normality & Central Limit Theorem

        PurchaseAmount approximately normal → justified parametric tests
      
        CLT validates reliability of sample mean for inference

5. Confidence Interval

        95% CI for average PurchaseAmount: (989.04, 1016.17)


🔍 Key Insights

  Average customer spending is stable (~1002.60)
  
  Spending differs significantly between genders, but not between campaigns or regions
  
  Product category does not significantly affect churn
  
  Outliers represent high-value customers and were retained for analysis
  
  Statistical tests and confidence intervals are valid due to approximate normality and large sample size


🚀 How to Run


    pip install pandas numpy scipy matplotlib
    
    python customer.py

👩‍💻
