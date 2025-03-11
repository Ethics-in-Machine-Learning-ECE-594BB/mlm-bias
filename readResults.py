import pickle

file_path = "./eval/bert-base-uncased"  # Replace with the actual model filename
with open(file_path, "rb") as f:
    data = pickle.load(f)

print(data['eval'])  # Prints the full dictionary of saved results
