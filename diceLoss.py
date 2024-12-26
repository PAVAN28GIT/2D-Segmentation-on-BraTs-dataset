import keras.backend as K

# For non-tumor and background region:
def dice_coef_0(y_true, y_pred, smooth=0.000001):
    y_true_f = K.flatten(y_true[:,:,:,0])
    y_pred_f = K.flatten(y_pred[:,:,:,0])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# For necrosis:
def dice_coef_1(y_true, y_pred, smooth=0.000001):
    y_true_f = K.flatten(y_true[:,:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,:,1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# For edema:
def dice_coef_2(y_true, y_pred, smooth=0.000001):
    y_true_f = K.flatten(y_true[:,:,:,2])
    y_pred_f = K.flatten(y_pred[:,:,:,2])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# For Enhancing Tumor (ET):
def dice_coef_4(y_true, y_pred, smooth=0.000001):
    y_true_f = K.flatten(y_true[:,:,:,4])
    y_pred_f = K.flatten(y_pred[:,:,:,4])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# The mean dice score is the mean of the dice scores of all the individual regions.
def dice_score(y_true, y_pred):
    d0 = dice_coef_0(y_true, y_pred, smooth=0.000001)
    d1 = dice_coef_1(y_true, y_pred, smooth=0.000001)
    d2 = dice_coef_2(y_true, y_pred, smooth=0.000001)
    d4 = dice_coef_4(y_true, y_pred, smooth=0.000001)

    dice_mean = (d0 + d1 + d2 + d4) / 4
    return dice_mean

# Dice Loss function, which is 1 minus the dice score
def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)
