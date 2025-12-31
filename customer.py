import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import matplotlib.pyplot as plt

df = pd.read_csv('customer_behavior.csv')
df.info()

# There are missing values in each column except 'CustomerID'. Hence, Missing values in categorical variables were
# replaced with a new category "Unknown" to retain information without introducing bias.
categorical_cols = ['Gender', 'Region', 'ProductCategory', 'CampaignGroup']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

# Numerical column 'PurchaseAmount'
# Median was used to impute missing purchase amounts as it is less sensitive to outliers.
df['PurchaseAmount'] = df['PurchaseAmount'].fillna(df['PurchaseAmount'].median())


# Rows with missing churn values were removed since churn is the target variable and cannot be reliably imputed.
df = df.dropna(subset=['Churn'])
print(df.isnull().sum())


# 1. What is the average, median, and mode of PurchaseAmount?

purchase_amount_mean = df['PurchaseAmount'].mean()
print(f"Average Purchase Amount: {purchase_amount_mean:.2f}")
purchase_amount_median = df['PurchaseAmount'].median()
print(f"Median Purchase Amount: {purchase_amount_median:.2f}")
purchase_amount_mode = df['PurchaseAmount'].mode()[0]
print(f"Mode Purchase Amount: {purchase_amount_mode:.2f}")
print("""The average purchase amount is approximately 1002.60, while the median and mode are both around 998.08.
Since the mean is slightly higher than the median, the distribution of purchase amounts is mildly right-skewed,
indicating the presence of a small number of high-spending customers. The equality of median and mode suggests
that most customers tend to spend around the same amount, reflecting consistent purchasing behavior.""")

# 2. Are there any outliers in the PurchaseAmount data?

Q1 = df['PurchaseAmount'].quantile(0.25)
Q3 = df['PurchaseAmount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['PurchaseAmount'] < lower_bound) | (df['PurchaseAmount'] > upper_bound)]
print("Number of outliers: ", outliers.shape[0])
print("""Outliers in the PurchaseAmount data were detected using the Inter quartile Range (IQR) method. A total of 22
 outliers were identified, indicating the presence of customers with unusually high or low purchase amounts. These 
 outliers explain the slight right-skewness observed in the data, where the mean is marginally higher than the median.
  Since these values likely represent high-value customers, they were retained for further analysis rather than being 
  removed.""")

plt.boxplot(df['PurchaseAmount'])
plt.title("Boxplot of Purchase Amount")
plt.ylabel("Purchase Amount")
plt.show()

# 3. Is there any skewness or kurtosis in the PurchaseAmount distribution?

skewness = df['PurchaseAmount'].skew()
kurtosis = df['PurchaseAmount'].kurt()
print("Skewness:", round(skewness, 2))
print("Kurtosis:", round(kurtosis, 2))
print("""The skewness of the PurchaseAmount distribution is 0.11, indicating a very slight positive skew and an almost
 symmetric distribution. This suggests that higher purchase values exist but do not heavily distort the overall 
 distribution. The kurtosis value is −0.18, which indicates slightly lighter tails than a normal distribution, 
 meaning that extreme purchase values are limited. Overall, the PurchaseAmount distribution is close to normal,
making it suitable for further statistical analysis.""")

# 4.Is there a significant difference in spending between male and female customers?

gender_df = df[df['Gender'].isin(['Male', 'Female'])]
male_purchase = gender_df[gender_df['Gender'] == 'Male']['PurchaseAmount']
female_purchase = gender_df[gender_df['Gender'] == 'Female']['PurchaseAmount']
t_stat, p_value = ttest_ind(male_purchase, female_purchase, equal_var=False)
print("T-statistic:", round(t_stat, 3))
print("P-value:", round(p_value, 4))
print("""An independent two-sample t-test was conducted to examine whether there is a significant difference in 
purchase amounts between male and female customers. The results showed a t-statistic of 2.229 and a p-value of 0.0259. 
Since the p-value is less than the significance level of 0.05, the null hypothesis was rejected. This indicates that, 
Yes,there is a statistically significant difference in spending between male and female 
customers.""")

# 5. Is there a relationship between ProductCategory and customer churn?

contingency_table = pd.crosstab(df['ProductCategory'], df['Churn'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print("Chi-square statistic:", round(chi2, 3))
print("P-value:", round(p_value, 4))
print("""A Chi-Square test of independence was conducted to examine the relationship between ProductCategory and customer
churn. The test produced a chi-square statistic of 2.693 and a p-value of 0.4415. Since the p-value is greater than the
significance level of 0.05, the null hypothesis was not rejected. This indicates that there is no statistically 
significant relationship between product category and customer churn, suggesting that churn behavior is independent of
the product category in this dataset.""")


# 6.Does PurchaseAmount vary significantly across different regions?

region_df = df[df['Region'] != 'Unknown']
groups = [
    region_df[region_df['Region'] == region]['PurchaseAmount']
    for region in region_df['Region'].unique()
]
f_stat, p_value = f_oneway(*groups)

print("F-statistic:", round(f_stat, 3))
print("P-value:", round(p_value, 4))
print("""A one-way ANOVA was performed to determine whether purchase amounts vary across different regions. The results
 showed an F-statistic of 0.381 and a p-value of 0.767. Since the p-value is greater than the significance level of 
 0.05, the null hypothesis was not rejected. This indicates that there is no statistically significant difference in 
 purchase amounts across regions, suggesting that regional location does not have a significant impact on customer 
 spending behavior.""")

# 7.Which email campaign (A or B) performed better in terms of average PurchaseAmount?

campaign_df = df[df['CampaignGroup'].isin(['A', 'B'])]
campaign_means = campaign_df.groupby('CampaignGroup')['PurchaseAmount'].mean().round(2)
A_purchase = campaign_df[campaign_df['CampaignGroup'] == 'A']['PurchaseAmount']
B_purchase = campaign_df[campaign_df['CampaignGroup'] == 'B']['PurchaseAmount']

t_stat, p_value = ttest_ind(A_purchase, B_purchase, equal_var=False)

print("T-statistic:", round(t_stat, 3))
print("P-value:", round(p_value, 4))
print("""To compare the performance of Email Campaign A and Campaign B, the average purchase amount for each campaign 
was analyzed. A two-sample t-test was conducted to assess whether the observed difference was statistically significant.
The results showed a t-statistic of 1.371 and a p-value of 0.1703. Since the p-value is greater than the significance 
level of 0.05, the null hypothesis was not rejected. This indicates that there is no statistically significant 
difference in average purchase amount between the two campaigns, and neither campaign can be considered superior in 
terms of customer spending.""")

# 8. Can we assume PurchaseAmount follows a normal distribution?

print("""Yes, PurchaseAmount can be reasonably assumed to follow an approximately normal distribution.
It is not perfectly normal, but close enough for practical analysis.Based on the descriptive and distributional 
analysis, PurchaseAmount can be reasonably assumed to follow an approximately normal distribution. The mean, median, and
mode are very close in value, indicating symmetry. The skewness value of 0.11 suggests a very mild right skew, while 
the kurtosis value of −0.18 indicates near-normal tail behavior. Although a small number of outliers are present, they
are not extreme. Therefore, the normality assumption is acceptable, and parametric statistical tests applied in the 
analysis are justified.""")


# 9.What insights can we gain by applying the Central Limit Theorem?

print("""By applying the Central Limit Theorem, we gain confidence that the distribution of the sample mean of 
PurchaseAmount is approximately normal, given the sufficiently large sample size. This holds true even if the individual
 purchase amounts are not perfectly normally distributed. As a result, the mean purchase amount serves as a reliable 
 measure of central tendency, and parametric statistical tests such as t-tests and ANOVA used in the analysis are 
 justified. The CLT thus enables valid statistical inference and reliable conclusions about customer spending behavior.
 """)

# 10.What is the 95% confidence interval for the average PurchaseAmount?

data = df['PurchaseAmount'].dropna()

mean = np.mean(data)
std = np.std(data, ddof=1)
n = len(data)

# 95% confidence interval
confidence_interval = stats.t.interval(
    0.95,
    df=n-1,
    loc=mean,
    scale=std / np.sqrt(n)
)
print("Mean:", round(mean, 2))
lower, upper = confidence_interval
print("Confidence Interval:", round(lower, 2), round(upper, 2))
print("""The average PurchaseAmount was found to be 1002.60. Using a 95% confidence level, the confidence interval 
for the mean PurchaseAmount was calculated as (989.04, 1016.17). This indicates that we are 95% confident that the 
true average purchase amount of customers lies within this range.""")



