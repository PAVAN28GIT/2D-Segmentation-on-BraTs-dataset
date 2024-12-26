from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load data and model
X_test = np.load("processed/X_test.npy")
Y_test = np.load("processed/Y_test.npy")
model = load_model("unet_best_model.h5")

# Evaluate the model
results = model.evaluate(X_test, Y_test, batch_size=4)
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])

# Predict on test data
predictions = model.predict(X_test)
predicted_masks = np.argmax(predictions, axis=-1)

# Visualize predictions
for i in range(10):  # Show 5 predictions
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(X_test[i, :, :, 0], cmap="gray")  # Show one modality (FLAIR)
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(np.argmax(Y_test[i], axis=-1), cmap="viridis")
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_masks[i], cmap="viridis")
    plt.show()
