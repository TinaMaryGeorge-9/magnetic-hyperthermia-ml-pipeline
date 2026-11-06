import pandas as pd
from src.model_engine import BioMachineLearningEngine

def run_research_pipeline():
    print("=== INITIALIZING MAGNETIC HYPERTHERMIA PREDICTIVE ENGINE ===\n")
    
    # 1. Set the column header name the AI will learn to predict
    target_label = "therapeutic_success" 
    
    try:
        # 2. Load your digitized lab matrix directly from the CSV file
        print("[*] Ingesting digitized experimental metrics from data/hyperthermia_dataset.csv...")
        df = pd.read_csv("data/hyperthermia_dataset.csv")
        
        # 3. Create a clean classification target (Binary Boundary)
        # Rule: If Optical Density drops below 0.08, cells died successfully (1). Otherwise (0).
        df[target_label] = (df['optical_density'] < 0.08).astype(int)
        
        # Drop the raw optical density numbers so the AI doesn't cheat by looking at the answer
        df = df.drop(columns=['optical_density'])
        
        # 4. Deploy the Machine Learning framework layers
        ml_system = BioMachineLearningEngine(random_state=42)
        ml_system.prepare_data(df=df, target_column=target_label)
        ml_system.train_model()
        ml_system.evaluate_performance()
        
        # 5. Extract and map feature weights
        feature_columns = list(df.drop(columns=[target_label]).columns)
        ml_system.plot_feature_importance(feature_names=feature_columns)
        
        print("\n=== PIPELINE RUN COMPLETE: ARCHITECTURAL PLOTS EXPORTED ===")
        
    except Exception as error:
        print(f"\n[!] Execution Interrupted: {error}")

if __name__ == "__main__":
    run_research_pipeline()
