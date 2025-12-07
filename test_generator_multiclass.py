from data_multiclass import get_multiclass_generators

print("Testing Stage-2 generator...")

train_gen, val_gen = get_multiclass_generators()

print("\nTrain generator loaded.")
print("Classes:", train_gen.class_indices)
print("Samples:", train_gen.samples)
print("Batch size:", train_gen.batch_size)
print("Number of batches per epoch:", len(train_gen))

# Try to fetch 1 batch
x, y = next(train_gen)

print("\nFetched 1 batch successfully!")
print("Batch X shape:", x.shape)
print("Batch Y shape:", y.shape)
