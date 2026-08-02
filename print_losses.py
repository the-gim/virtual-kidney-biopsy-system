import joblib
import os
import matplotlib.pyplot as plt

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
model_path = os.path.join(DATA_DIR, "artificial_neural_network.joblib")
model = joblib.load(model_path)
mlp = model.named_steps['mlp']
print("Train losses (first 10, last 10):")
print(mlp.manual_train_loss_[:10])
print(mlp.manual_train_loss_[-10:])
print("Val losses (first 10, last 10):")
print(mlp.manual_val_loss_[:10])
print(mlp.manual_val_loss_[-10:])
