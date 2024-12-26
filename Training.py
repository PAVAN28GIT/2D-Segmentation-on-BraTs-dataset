import os
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from unetModel import UNet
from diceLoss import dice_loss, dice_coef_0, dice_coef_1, dice_coef_2, dice_coef_4, dice_score


# Define the target shape
target_shape = (240, 240, 4)

# Load all image-mask pairs
image_files = [f for f in os.listdir('processedData') if f.endswith('_images.npy')]
mask_files = [f for f in os.listdir('processedData') if f.endswith('_masks.npy')]

image_files.sort()
mask_files.sort()

print("Number of images:", len(image_files))
print("Number of masks:", len(mask_files))

print("Loading images and masks...")
# Load and resize images
images_resized = []
for image_file in image_files:
    img = np.load(os.path.join('processedData', image_file))
    img_resized = resize(img, target_shape, mode='constant', preserve_range=True)
    images_resized.append(img_resized)

print("Images loaded and resized.")

# Load and resize masks
masks_resized = []
for mask_file in mask_files:
    mask = np.load(os.path.join('processedData', mask_file))
    # Resize masks, but typically masks are 2D so we keep only 2 dimensions
    mask_resized = resize(mask, target_shape[:2], mode='constant', preserve_range=True)
    masks_resized.append(mask_resized)

print("Masks loaded and resized.")

# Convert to numpy arrays
images_resized = np.array(images_resized)
masks_resized = np.array(masks_resized)

print("Images shape:", images_resized.shape)
print("Masks shape:", masks_resized.shape)

# Split data into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(images_resized, masks_resized, test_size=0.2, random_state=42)

print("Training data shapes:")
print(X_train.shape)
print(Y_train.shape)

# Load the saved train and test splits
X_train = np.load('X_train.npy')
X_valid = np.load('X_valid.npy')
Y_train = np.load('Y_train.npy')
Y_valid = np.load('Y_valid.npy')

# Define the model
model = UNet()  
model.compile(optimizer=Adam(lr=0.0001), loss=dice_loss,
              metrics=[dice_coef_0, dice_coef_1, dice_coef_2, dice_coef_4, dice_score])

# Define the checkpoint to save the best model
checkpoint = ModelCheckpoint('Unet_checkpoints/weights-improvement-{epoch:02d}-{dice_score:.3f}.h5',
                             monitor='val_dice_score', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

# Training the model
batch_size = 8  
epochs = 5 

# Adding print statements to ensure visibility
print("Training started...")

history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
                    batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks_list)

print("Training finished.")

# Optional: Print the final model performance
print("Final model metrics:", history.history)
