import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, \
accuracy_score
from sklearn.feature_selection import RFE

"""
Reading in Data
"""

data = pd.read_csv('alzheimers_disease_data.csv')
data.head()
data.describe()

columns = ['BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', \
'DietQuality', 'SleepQuality', 'Diagnosis']
df = data[columns]

"""
Data Visualizations
"""

print(df)

plot.title('Distribution of Patient Diagnosis')
sns.distplot(data['Diagnosis'])
plot.show()

sns.boxplot(x = 'Diagnosis', y = 'BMI', data = df)
plot.title('Correlation between BMI and Alzheimers Diagnosis')
plot.ylabel('BMI')
plot.xlabel('Alzheimers Diagnosis')
plot.show()

sns.barplot(x = 'Diagnosis', y = 'Smoking', errorbar = None, data = df)
plot.title('Correlation between Smoking and Alzheimers Diagnosis')
plot.ylabel('Smoking')
plot.xlabel('Alzheimers Diagnosis')
plot.show()

sns.boxplot(x = 'Diagnosis', y = 'AlcoholConsumption', data = df)
plot.title('Correlation between AlcoholConsumption and Alzheimers Diagnosis')
plot.ylabel('AlcoholConsumption')
plot.xlabel('Alzheimers Diagnosis')
plot.show()

sns.boxplot(x = 'Diagnosis', y = 'PhysicalActivity', data = df)
plot.title('Correlation between Physical Activity and Alzheimers Diagnosis')
plot.ylabel('Physical Activity')
plot.xlabel('Alzheimers Diagnosis')
plot.show()

sns.boxplot(x = 'Diagnosis', y = 'DietQuality', data = df)
plot.title('Correlation between Diet Quality and Alzheimers Diagnosis')
plot.ylabel('Diet Quality')
plot.xlabel('Alzheimers Diagnosis')
plot.show()

sns.boxplot(x = 'Diagnosis', y = 'SleepQuality', data = df)
plot.title('Correlation between Sleep Quality and Alzheimers Diagnosis')
plot.ylabel('Sleep Quality')
plot.xlabel('Alzheimers Diagnosis')
plot.show()

"""
Splitting into Train and Test Data
"""

x = data[columns[:-1]]
y = data[columns[-1]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, \
random_state=0)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

print(x)
print(y)

"""
Training Logistic Regression Model
"""

log_reg = LogisticRegression(class_weight='balanced')
# log_reg = LogisticRegression()
rfe = RFE(log_reg, n_features_to_select=5).fit(x_train_scaled,y_train)
print("Feature Ranking:", rfe.ranking_)
selected_features = x.columns[rfe.support_]
print("Selected Features:", selected_features)

y_pred = rfe.predict(x_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

"""
Training Random Forest Classifier Model
"""

random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(x_train, y_train)
feature_importances = random_forest.feature_importances_
features = x.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': \
feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)