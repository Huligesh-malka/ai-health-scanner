import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("eye_predictions.csv")

# Count 'yes' and 'no' predictions
yes_count = (df['predict_face'] == 'yes').sum()

print(f"Yes predictions: {yes_count}")
print(f"No predictions: {no_count}")

# Visualize prediction counts
plt.bar(['Yes', 'No'], [yes_count, no_count], color=['green', 'red'])
plt.title('Prediction Counts')
plt.ylabel('Number of Predictions')
plt.show()

# Compare prediction to true labels, if available
if 'true_label' in df.columns:
    correct = (df['prediction'] == df['true_label']).sum()
    total = len(df)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
else:
    print("Column 'true_label' not found. Skipping accuracy calculation.")

# Optional: Save results to new file
df['correct'] = df['prediction'] == df.get('true_label')
df.to_csv("eye_predictions_with_eval.csv", index=False)
print("Results saved to eye_predictions_with_eval.csv")