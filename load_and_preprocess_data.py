import os
import numpy as np
import nibabel as nib

# Constants
IMG_SIZE = (240, 240)  # Height and width of images
N_CLASSES = 4  # Segmentation classes: 0, 1, 2, 3
MODALITIES = ['flair', 't1', 't1ce', 't2']  # List of all modalities


def load_nifti_file(filepath):
    """Load a NIfTI file and return its data as a NumPy array."""
    nifti_img = nib.load(filepath)
    return np.asarray(nifti_img.get_fdata(), dtype=np.float32)


def normalize_image(image):
    """Normalize image intensity values to range [0, 1]."""
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)


def preprocess_patient_data(patient_path, modalities, segmentation):
    """Load and preprocess data for a single patient."""
    # Load all modalities
    modality_slices = []
    for modality in modalities:
        modality_path = os.path.join(patient_path, f"{os.path.basename(patient_path)}_{modality}.nii")
        modality_data = load_nifti_file(modality_path)
        modality_data = normalize_image(modality_data)
        modality_slices.append(modality_data)

    # Stack modalities to create a multi-channel image
    multi_channel_data = np.stack(modality_slices, axis=-1)  # Shape: (H, W, D, 4)

    # Load segmentation mask
    seg_path = os.path.join(patient_path, f"{os.path.basename(patient_path)}_{segmentation}.nii")
    seg_data = load_nifti_file(seg_path)

    # Convert mask values (BraTS labels: 1, 2, 4 -> 1, 2, 3)
    seg_data[seg_data == 4] = 3

    # Slice 3D volume into 2D images
    images = []
    masks = []
    for slice_idx in range(multi_channel_data.shape[2]):  # Iterate over axial slices
        image_slice = multi_channel_data[:, :, slice_idx, :]
        mask_slice = seg_data[:, :, slice_idx]
        if np.sum(mask_slice) == 0:  # Skip empty slices
            continue
        images.append(image_slice)
        masks.append(mask_slice)

    return np.array(images), np.array(masks)


def preprocess_and_save_dataset(dataset_path, output_dir, dataset_type, modalities, segmentation='seg'):
    """Preprocess the entire dataset (training or validation) and save incrementally."""
    dataset_output_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_output_dir, exist_ok=True)

    print(f"Processing {dataset_type} dataset...")
    for patient in os.listdir(dataset_path):
        patient_path = os.path.join(dataset_path, patient)
        if not os.path.isdir(patient_path):
            continue

        print(f"Processing patient: {patient}")
        # Preprocess data for each patient
        images, masks = preprocess_patient_data(patient_path, modalities, segmentation)

        # Save each patient's data incrementally
        patient_images_path = os.path.join(dataset_output_dir, f"{patient}_images.npy")
        patient_masks_path = os.path.join(dataset_output_dir, f"{patient}_masks.npy")

        np.save(patient_images_path, images)
        np.save(patient_masks_path, masks)

        print(f"Saved images to {patient_images_path}")
        print(f"Saved masks to {patient_masks_path}")


# Example usage
if __name__ == "__main__":
    # Paths
    TRAINING_DATA_PATH = "BraTs2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    VALIDATION_DATA_PATH = "BraTs2020_ValidationData/MICCAI_BraTS2020_ValidationData"
    OUTPUT_DIR = "preprocessed_data"

    # Preprocess and save training data
    preprocess_and_save_dataset(TRAINING_DATA_PATH, OUTPUT_DIR, "train", MODALITIES)
