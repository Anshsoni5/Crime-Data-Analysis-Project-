# 🔹 Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 🔹 Load Dataset
df = pd.read_csv("C:/Users/Asus/Downloads/crimedata.csv")

# 🔹 Clean Columns
df.columns = df.columns.str.strip()
df.columns = ['Country', 'State', 'Year', 'Category', 'Sub_Category', 'Incidence', 'Victims', 'Crime_Rate']

# 🔹 Convert Numeric
df['Incidence'] = pd.to_numeric(df['Incidence'], errors='coerce')
df['Victims'] = pd.to_numeric(df['Victims'], errors='coerce')

# 🔹 Drop Missing
df = df.dropna()

# =====================================================
# 📊 OBJECTIVE 1: Top 10 States by Crime
# =====================================================

top_states = df.groupby('State')['Incidence'].sum().sort_values(ascending=False).head(10)

plt.figure()
top_states.plot(kind='bar')
plt.title("Top 10 States by Crime")
plt.xlabel("State")
plt.ylabel("Total Crime")
plt.xticks(rotation=45)
plt.show()

# =====================================================
# 📊 OBJECTIVE 2: Year-wise Crime Trend
# =====================================================

yearly = df.groupby('Year')['Incidence'].sum()

plt.figure()
yearly.plot(kind='line')
plt.title("Year-wise Crime Trend")
plt.xlabel("Year")
plt.ylabel("Total Crime")
plt.show()

# =====================================================
# 📊 OBJECTIVE 3: Crime Category Distribution
# =====================================================

top_category = df.groupby('Category')['Incidence'].sum().sort_values(ascending=False).head(10)

plt.figure()
top_category.plot(kind='bar')
plt.title("Top Crime Categories")
plt.xlabel("Category")
plt.ylabel("Total Crime")
plt.xticks(rotation=45)
plt.show()

# =====================================================
# 📊 OBJECTIVE 4: Incidence vs Victims (Relationship)
# =====================================================

plt.figure()
plt.scatter(df['Incidence'], df['Victims'])
plt.title("Incidence vs Victims")
plt.xlabel("Incidence")
plt.ylabel("Victims")
plt.show()

# =====================================================
# 📊 OBJECTIVE 5: Crime Distribution (Histogram)
# =====================================================

plt.figure()
df['Incidence'].plot(kind='hist', bins=20)
plt.title("Crime Distribution")
plt.xlabel("Incidence")
plt.ylabel("Frequency")
plt.show()

# =====================================================
# 📊 HEATMAP
# =====================================================

plt.figure()
sns.heatmap(df[['Incidence','Victims','Crime_Rate']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# =====================================================
# 📊 OUTLIERS (IQR METHOD)
# =====================================================

Q1 = df['Incidence'].quantile(0.25)
Q3 = df['Incidence'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['Incidence'] < (Q1 - 1.5*IQR)) | (df['Incidence'] > (Q3 + 1.5*IQR))]

print("\nOutliers:\n", outliers)

plt.figure()
sns.boxplot(x=df['Incidence'])
plt.title("Outlier Detection")
plt.show()

# =====================================================
# 📊 Z-TEST
# =====================================================

sample = df['Incidence']
pop_mean = 1000

z_score = (np.mean(sample) - pop_mean) / (np.std(sample) / np.sqrt(len(sample)))

print("\nZ-Score:", z_score)

# =====================================================
# 📊 T-TEST
# =====================================================

t_stat, p_value = stats.ttest_1samp(sample, pop_mean)

print("\nT-Statistic:", t_stat)
print("P-Value:", p_value)

# =====================================================
# 🤖 LINEAR REGRESSION (ONLY MODEL)
# =====================================================

X = df[['Incidence']]
y = df['Victims']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n----- LINEAR REGRESSION -----")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# =====================================================
# 📈 REGRESSION GRAPH
# =====================================================

plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Linear Regression Line")
plt.xlabel("Incidence")
plt.ylabel("Victims")
plt.show()