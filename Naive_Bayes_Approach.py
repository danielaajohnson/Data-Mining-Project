""" NAIVE BAYES MODEL """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Importing train dataset
df_train = pd.read_csv("./census-income.data.csv")
print(df_train.head())


duplicate_rows = df_train.duplicated().sum()
print("Number of duplicate rows:", duplicate_rows)


# Dropping duplicate rows
df_train.drop_duplicates(inplace=True)

# Confirming that duplicate rows are removed
print("Number of duplicate rows left:", df_train.duplicated().sum())


# Importing test dataset
df_test = pd.read_csv("./census-income.test.csv")
print(df_test.head())


# Remove dots from the column 'income' from test dataset
df_test['income'] = df_test['income'].str.replace('.', '')
print(df_test.head())


# Fill missing values for the train data
df_train['work-class'] = df_train['work-class'].replace(' ?', df_train['work-class'].mode()[0])

df_train['occupation'] = df_train['occupation'].replace(' ?', df_train['occupation'].mode()[0])

df_train['native-country'] = df_train['native-country'].replace(' ?', df_train['native-country'].mode()[0])

# Fill missing values for the test data
df_test['work-class'] = df_test['work-class'].replace(' ?', df_test['work-class'].mode()[0])

df_test['occupation'] = df_test['occupation'].replace(' ?', df_test['occupation'].mode()[0])

df_test['native-country'] = df_test['native-country'].replace(' ?', df_test['native-country'].mode()[0])

# Counting columns values
df_train['income'].value_counts()


df_test['income'].value_counts()


df_train['native-country'].value_counts()

# Creating new train DF without columns education and fnlwgt
df_train_c = df_train.drop(['education', 'fnlwgt'], axis = 1)
df_train_c.head(5)

# Creating new test DF without columns education and fnlwgt
df_test_c = df_test.drop(['education', 'fnlwgt'], axis = 1)
df_test_c.head(5)

# Encoding categorical values into numerical values train data
en = LabelEncoder()
df_train_c['work-class'] = en.fit_transform(df_train_c['work-class'])
df_train_c['marital-status'] = en.fit_transform(df_train_c['marital-status'])
df_train_c['occupation'] = en.fit_transform(df_train_c['occupation'])
df_train_c['relationship'] = en.fit_transform(df_train_c['relationship'])
df_train_c['race'] = en.fit_transform(df_train_c['race'])
df_train_c['sex'] = en.fit_transform(df_train_c['sex'])
df_train_c['native-country'] = en.fit_transform(df_train_c['native-country'])
df_train_c['income'] = en.fit_transform(df_train_c['income'])

df_train_c.head()

#Encoding categorical values into numerical values test data
en = LabelEncoder() 
df_test_c['work-class'] = en.fit_transform(df_test_c['work-class'])
df_test_c['marital-status'] = en.fit_transform(df_test_c['marital-status'])
df_test_c['occupation'] = en.fit_transform(df_test_c['occupation'])
df_test_c['relationship'] = en.fit_transform(df_test_c['relationship'])
df_test_c['race'] = en.fit_transform(df_test_c['race'])
df_test_c['sex'] = en.fit_transform(df_test_c['sex'])
df_test_c['native-country'] = en.fit_transform(df_test_c['native-country'])
df_test_c['income'] = en.fit_transform(df_test_c['income'])

df_test_c.head()

# Train & Test Data Income Distribution
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Train data
axs[0].bar(df_train['income'].value_counts().index, df_train['income'].value_counts(), color='skyblue')
axs[0].set_title('Train Income Distribution')
axs[0].set_xlabel('Income')
axs[0].set_ylabel('Count')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# Test data
axs[1].bar(df_test['income'].value_counts().index, df_test['income'].value_counts(), color='skyblue')
axs[1].set_title('Test Income Distribution')
axs[1].set_xlabel('Income')
axs[1].set_ylabel('Count')
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#Plotting top 5 education_num and nums
df_high_income = df_train_c[df_train_c['income'] == 1]

top_education_levels = df_high_income['education_num'].value_counts().head(5)
top_education_levelsB = df_high_income['education_num'].value_counts()
plt.figure(figsize=(12, 6))

# Define shades of blue color palette
blue_palette = ['#0072B2', '#4C92C3', '#6BAED6', '#9ECAE1', '#C6DBEF']

# Create pie chart
plt.subplot(1, 2, 1)
plt.pie(top_education_levels, labels=top_education_levels.index, autopct='%1.1f%%', startangle=140, colors=blue_palette)
plt.title('Top 5 Education Levels with Income > $50k')
plt.axis('equal')

plt.subplot(1, 2, 2)
bars = plt.bar(top_education_levelsB.index.astype(str), top_education_levelsB.values, color='skyblue')
plt.title('Count of Each Education Level >$50k')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), va='bottom')

plt.show()

# Plotting Heatmap
sns.heatmap(df_train_c.corr(), cmap="Blues", annot=True, fmt=".2f", annot_kws={"size": 5})

plt.show()

# Splitting data into test and train
X_train = df_train_c.drop('income', axis=1)
y_train = df_train_c['income']

X_test = df_test_c.drop('income', axis=1)
y_test = df_test_c['income']

# Importing Naive Bayes
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X_train,y_train)

# Importing Metrics Libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Accuracy for test data
print("\n Accuracy for the Test Data\n")
labels = ['<=50K', '>50K']
y_pred = gb.predict(X_test)

print(classification_report(y_test, y_pred, target_names=labels))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
print(conf_matrix_df)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Accuracy for train data
labels = ['<=50K', '>50K']
y_pred_train = gb.predict(X_train)

print("\nAccuracy for Train Data:\n")
print(classification_report(y_train, y_pred_train, target_names=labels))

print("Confusion Matrix for train data:")
conf_matrix_train = confusion_matrix(y_train, y_pred_train)
conf_matrix_df_train = pd.DataFrame(conf_matrix_train, index=labels, columns=labels)
print(conf_matrix_df_train)

print(f"Accuracy for train data: {accuracy_score(y_train, y_pred_train) * 100:.2f}%")

# Laplace Smoothing test data

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=1.0)

# Train the model
mnb.fit(X_train, y_train)

# Predict using the trained model
y_predLS = mnb.predict(X_test)
print("\nPerformance for Test Data W/ Laplace Smoothing")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_predLS, target_names=labels))
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_predLS)
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
print(conf_matrix_df)
print(f"Accuracy: {accuracy_score(y_test, y_predLS) * 100:.2f}%")


# Predict using the trained model on training data
y_pred_train_LS = mnb.predict(X_train)

# Evaluate performance on training data
print("\nPerformance on Train Data W/ Laplace Smoothing:")
print(classification_report(y_train, y_pred_train_LS, target_names=labels))
print("Confusion Matrix:")
conf_matrix_train = confusion_matrix(y_train, y_pred_train_LS)
conf_matrix_train_df = pd.DataFrame(conf_matrix_train, index=labels, columns=labels)
print(conf_matrix_train_df)
print(f"Accuracy: {accuracy_score(y_train, y_pred_train_LS) * 100:.2f}%")



