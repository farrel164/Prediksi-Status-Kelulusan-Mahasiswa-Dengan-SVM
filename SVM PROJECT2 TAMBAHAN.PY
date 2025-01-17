import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load dataset
studata = pd.read_csv(r"C:\Users\Farrel\Downloads\dataset500.csv")
filtered_data = studata[studata['Target'].isin(['Dropout', 'Graduate'])]
filtered_data_encoded = filtered_data.copy()
filtered_data_encoded['Target'] = filtered_data_encoded['Target'].map({'Graduate': 1, 'Dropout': 0})

# Splitting features and target
x = filtered_data_encoded.drop(columns=["Target"])
y = filtered_data_encoded["Target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# PCA for dimensionality reduction
pca = PCA(n_components=0.60)  # Retain 60% of the variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"\nOriginal number of features: {X_train.shape[1]}")
print(f"Reduced number of features after PCA: {X_train_pca.shape[1]}")

# Model building with PCA
stumodel = SVC(kernel='rbf', class_weight='balanced', random_state=42)

# Stratified K-Fold Cross-Validation with PCA-transformed data
strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 10 folds

cv_scores = cross_val_score(stumodel, X_train_pca, y_train, cv=strat_kfold, scoring='accuracy')

print("\nStratified K-Fold Cross-Validation Results:")
print(f"Mean Stratified K-Fold Accuracy: {cv_scores.mean():.2f}")
print(f"Standard Deviation of Accuracy: {cv_scores.std():.2f}")

# Train and evaluate the model
stumodel.fit(X_train_pca, y_train)
y_pred = stumodel.predict(X_test_pca)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Metrics for both classes: Dropout (0) and Graduate (1)

# Precision, Recall, and F1 Score for Dropout (0)
precision_drop = precision_score(y_test, y_pred, pos_label=0)
recall_drop = recall_score(y_test, y_pred, pos_label=0)
f1_drop = f1_score(y_test, y_pred, pos_label=0)

# Precision, Recall, and F1 Score for Graduate (1)
precision_grad = precision_score(y_test, y_pred, pos_label=1)
recall_grad = recall_score(y_test, y_pred, pos_label=1)
f1_grad = f1_score(y_test, y_pred, pos_label=1)

# Print the metrics
print(f"\nPrecision for Dropout (0): {precision_drop:.2f}")
print(f"Recall for Dropout (0): {recall_drop:.2f}")
print(f"F1 Score for Dropout (0): {f1_drop:.2f}")

print(f"\nPrecision for Graduate (1): {precision_grad:.2f}")
print(f"Recall for Graduate (1): {recall_grad:.2f}")
print(f"F1 Score for Graduate (1): {f1_grad:.2f}")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Dropout', 'Graduate'],
            yticklabels=['Dropout', 'Graduate'])
plt.title('Confusion Matrix (Dropout and Graduate)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nExplained Variance Ratio by PCA Components:")
print(pca.explained_variance_ratio_)

# Prediction function for user input
def predict_user_input():
    field_inputs = {}
    field_inputs['Course'] = int(input("Enter Course (1-17): "))
    field_inputs['Tuition fees up to date'] = int(input("Is tuition up to date? (0=No, 1=Yes): "))
    field_inputs['Gender'] = int(input("Enter Gender (0=Female, 1=Male): "))
    field_inputs['Scholarship holder'] = int(input("Is the student a scholarship holder? (0=No, 1=Yes): "))
    field_inputs['Curricular units 1st sem (approved)'] = int(input("Enter approved courses in 1st semester: "))
    field_inputs['Curricular units 1st sem (grade)'] = float(input("Enter average grade in 1st semester: "))
    field_inputs['Curricular units 2nd sem (approved)'] = int(input("Enter approved courses in 2nd semester: "))
    field_inputs['Curricular units 2nd sem (grade)'] = float(input("Enter average grade in 2nd semester: "))
    
    new_data_df = pd.DataFrame([field_inputs])
    new_data_scaled = scaler.transform(new_data_df)  # Standardize
    new_data_pca = pca.transform(new_data_scaled)  # Apply PCA
    prediction = stumodel.predict(new_data_pca)
    target_labels = {0: 'Dropout', 1: 'Graduate'}
    print("\nPrediction for the new data:", target_labels[prediction[0]])

predict_user_input()
