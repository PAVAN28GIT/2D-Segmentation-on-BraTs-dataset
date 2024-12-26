from keras.layers import Conv2D, Input, Dropout, MaxPooling2D, UpSampling2D, Conv2DTranspose, Concatenate, BatchNormalization
from keras.models import Model

# U-net inputs 2D images of size 240x240 with 4 channels (flair, t1, t1ce, and t2 modalities) and outputs segmentation masks with 5 classes (background, necrosis, edema, non-enhancing tumor, and enhancing tumor).

def UNet(n_classes=5, im_sz=240, n_channels=4, n_filters_start=64, growth_factor=2):
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    
    # Encoder
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    n_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    n_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    n_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)

    # Decoder
    n_filters //= growth_factor # filters are halved 
    up6 = Concatenate()([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    up6 = BatchNormalization()(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6)

    n_filters //= growth_factor
    up7 = Concatenate()([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)

    n_filters //= growth_factor
    up8 = Concatenate()([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)

    n_filters //= growth_factor
    up9 = Concatenate()([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

    # Output layer
    conv10 = Conv2D(n_classes, (1, 1), activation='softmax')(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    return model

