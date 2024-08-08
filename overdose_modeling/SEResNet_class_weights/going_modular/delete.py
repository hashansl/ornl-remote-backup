import pandas as pd

annotation_file_path = "/home/h6x/Projects/overdose_modeling/data/processed_data/svi_with_hepvu/2018/annotations_2018/annotation_NOD.csv"
root_dir = "/home/h6x/Projects/data_processing/data/processed_data/persistence_images/below_90th/h0h1/different_kernel_spread/npy_combined_features"

dtype = {'STCNTY': str}

annotations = pd.read_csv(annotation_file_path,dtype=dtype)

print(annotations.head())
print(annotations['STCNTY'][0])
print(type(annotations['STCNTY'][0]))