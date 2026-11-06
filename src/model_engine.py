import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class BioMachineLearningEngine:
    """Predictive analysis framework evaluating interdisciplinary feature importance weights."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def prepare_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2):
        """Splits experimental features into training arrays and validation parameters."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Stratifying preserves identical class boundaries across splits
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        print(f"[+] Dataset partitioned: {self.X_train.shape[0]} train rows, {self.X_test.shape[0]} validation rows.")

    def train_model(self):
        """Trains the model by optimizing split criteria over random feature sub-spaces."""
        print("[*] Training Random Forest ensemble classifier...")
        self.model.fit(self.X_train, self.y_train)
        print("[+] Ensemble learning optimization complete.")

    def evaluate_performance(self):
        """Validates model accuracy against unseen test targets."""
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        
        print("\n================ ARCHITECTURAL PERFORMANCE EVALUATION ================")
        print(f"Overall Model Classification Accuracy: {accuracy * 100:.2f}%")
        print("\nDetailed Performance Diagnostics:")
        print(classification_report(self.y_test, predictions))
        print("======================================================================")

    def plot_feature_importance(self, feature_names: list):
        """Maps relative feature weights to isolate which lab variables influence decisions most."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="mako")
        plt.title("Statistical Feature Weight Analysis (Research Mapping)")
        plt.xlabel("Mathematical Target Gini Importance Weight")
        plt.ylabel("Laboratory Experimental Metrics")
        plt.tight_layout()
        
        plt.savefig("feature_importance.png")
        print("[+] Relative statistical graph exported locally as 'feature_importance.png'.")
