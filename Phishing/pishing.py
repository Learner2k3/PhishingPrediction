import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the dataset from the provided CSV file
data = pd.read_csv('Phishing.csv')

# Separate features (X) and labels (y)
X = data.drop(columns=['id', 'CLASS_LABEL'])
y = data['CLASS_LABEL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a Random Forest classifier (you can choose another algorithm if desired)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save the trained model to a file
joblib.dump(model, 'phishing_model.pkl')


# Make predictions with the trained model
def predict_phishing(input_data):
    # Convert input_data dictionary to a DataFrame with the same columns as X_train
    input_df = pd.DataFrame([input_data])

    # Predict using the trained model
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        return "Phishing"
    else:
        return "Not Phishing"


# Example usage:
input_url_data = {
    'NumDots': 3,
    'SubdomainLevel': 1,
    'PathLevel': 5,
    'UrlLength': 72,
    'NumDash': 0,
    'NumDashInHostname': 0,
    'AtSymbol': 0,
    'TildeSymbol': 0,
    'NumUnderscore': 0,
    'NumPercent': 0,
    'NumQueryComponents': 0,
    'NumAmpersand': 0,
    'NumHash': 0,
    'NumNumericChars': 0,
    'NoHttps': 1,
    'RandomString': 1,
    'IpAddress': 0,
    'DomainInSubdomains': 0,
    'DomainInPaths': 0,
    'HttpsInHostname': 0,
    'HostnameLength': 21,
    'PathLength': 44,
    'QueryLength': 0,
    'DoubleSlashInPath': 0,
    'NumSensitiveWords': 0,
    'EmbeddedBrandName': 0,
    'PctExtHyperlinks': 0.25,
    'PctExtResourceUrls': 1,
    'ExtFavicon': 1,
    'InsecureForms': 0,
    'RelativeFormAction': 0,
    'ExtFormAction': 0,
    'AbnormalFormAction': 0,
    'PctNullSelfRedirectHyperlinks': 0,
    'FrequentDomainNameMismatch': 0,
    'FakeLinkInStatusBar': 0,
    'RightClickDisabled': 0,
    'PopUpWindow': 0,
    'SubmitInfoToEmail': 0,
    'IframeOrFrame': 1,
    'MissingTitle': -1,
    'ImagesOnlyInForm': 1,
    'SubdomainLevelRT': 1,
    'UrlLengthRT': 0,
    'PctExtResourceUrlsRT': 1,
    'AbnormalExtFormActionR': 1,
    'ExtMetaScriptLinkRT': -1,
    'PctExtNullSelfRedirectHyperlinksRT': 1
}

result = predict_phishing(input_url_data)
print(f'The input URL data is classified as: {result}')
