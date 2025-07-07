# Notebook 1: sales_forecasting/sales_analysis.ipynb

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Generate Synthetic Data
np.random.seed(0)
data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=200),
    'products_sold': np.random.randint(20, 100, size=200),
    'marketing_spend': np.random.randint(500, 2000, size=200)
})
data['sales'] = data['products_sold'] * 15 + data['marketing_spend'] * 0.3 + np.random.normal(0, 100, size=200)

# Step 3: Data Cleaning
# (Synthetic data, so minimal cleaning needed)
data.dropna(inplace=True)

# Step 4: Regression Analysis
X = data[['products_sold', 'marketing_spend']]
y = data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 5: Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Step 6: Visualization
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()


# Notebook 2: quality_control/manufacturing_quality.ipynb

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway

# Step 2: Generate Synthetic Data
np.random.seed(42)
data = pd.DataFrame({
    'production_time': np.random.normal(5, 1.5, 200),
    'temperature': np.random.normal(70, 5, 200),
    'humidity': np.random.uniform(30, 70, 200),
    'quality_score': np.random.normal(80, 10, 200)
})

# Step 3: ANOVA (Example: Check effect of temperature grouping)
data['temp_group'] = pd.cut(data['temperature'], bins=3, labels=['Low', 'Medium', 'High'])
anova_result = f_oneway(
    data[data['temp_group']=='Low']['quality_score'],
    data[data['temp_group']=='Medium']['quality_score'],
    data[data['temp_group']=='High']['quality_score']
)
print("ANOVA result:", anova_result)

# Step 4: T-Test (Compare two humidity levels)
data['humidity_group'] = pd.cut(data['humidity'], bins=2, labels=['Low', 'High'])
ttest_result = ttest_ind(
    data[data['humidity_group']=='Low']['quality_score'],
    data[data['humidity_group']=='High']['quality_score']
)
print("T-test result:", ttest_result)

# Step 5: Visualization
sns.boxplot(x='temp_group', y='quality_score', data=data)
plt.title("Quality Score by Temperature Group")
plt.show()


# Notebook 3: fraud_detection/fraud_detection.ipynb

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Step 2: Generate Synthetic Transaction Data
np.random.seed(99)
data = pd.DataFrame({
    'amount': np.concatenate([np.random.normal(50, 10, 190), np.random.normal(300, 20, 10)]),
    'transaction_type': np.random.choice(['online', 'offline'], 200)
})
data['type_code'] = data['transaction_type'].map({'offline': 0, 'online': 1})

# Step 3: Isolation Forest for Anomaly Detection
clf = IsolationForest(contamination=0.05, random_state=99)
data['anomaly_score'] = clf.fit_predict(data[['amount', 'type_code']])
data['anomaly'] = data['anomaly_score'] == -1

# Step 4: Visualization
plt.figure(figsize=(10,6))
plt.scatter(data.index, data['amount'], c=data['anomaly'], cmap='coolwarm')
plt.title("Anomaly Detection in Transaction Amounts")
plt.xlabel("Transaction Index")
plt.ylabel("Amount")
plt.show()

print("Total anomalies detected:", data['anomaly'].sum())
