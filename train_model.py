import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

def load_data(file_path):
    """Load and preprocess the credit card dataset"""
    df = pd.read_csv(file_path)
    return df

def train_model(df, use_top_features=True):
    """Train the Random Forest model and return the trained model and scaler"""
    # Handle missing values if any
    df = df.fillna(df.mean())
    
    # Define the specific features we want to use
    required_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']
    
    # Verify all required features exist in the dataset
    missing_features = [feature for feature in required_features if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in dataset: {missing_features}")
    
    # Filter dataframe to keep only required features
    df_filtered = df[required_features]
    
    # Split features and target
    X = df_filtered.drop('Class', axis=1)
    y = df_filtered['Class']
    
    # Get feature names for later use
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
    
    # Measure training time
    start_time = time.time()
    
    # Train Random Forest model
    model = RandomForestClassifier(
    n_estimators=200,         # slightly more trees
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',  # NEW: handle fraud imbalance
    max_depth=10,             # NEW: prevent overfitting
    min_samples_split=5,      # NEW: better generalization
    min_samples_leaf=2        # NEW: better handling of small classes
)

    model.fit(X_train, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'training_time': training_time
    }

def save_model(model_data, model_path='model.joblib', scaler_path='scaler.joblib'):
    """Save the trained model and scaler to disk"""
    joblib.dump(model_data['model'], model_path)
    joblib.dump(model_data['scaler'], scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    # Load the dataset
    df = load_data('creditcard.csv')  # Replace with your dataset path
    
    # Train the model
    model_data = train_model(df, use_top_features=True)
    
    # Save the model and scaler
    save_model(model_data)
    
    # Print some metrics
    print(f"Model accuracy: {model_data['accuracy']:.4f}")
    print(f"Training time: {model_data['training_time']:.2f} seconds")