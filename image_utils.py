import matplotlib.pyplot as plt
import numpy as np

from scipy import interpolate

def align_UKBvolume(input_vol, affine_matrix=None):
    "input UKBScan volume like: 204 x 208 x 13, output 13 x 208 x 204"
    output = np.flip(input_vol.squeeze().transpose((2, 1, 0)), axis=0)

    if affine_matrix is None:
        return output
    else:
        R, b = affine_matrix[0:3, 0:3], affine_matrix[0:3, -1]
        T = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        bT = np.array([output.shape[0]-1, 0, 0])

        R_new = R.dot(T)
        b_new = - R.dot(T).dot(bT) + b
        new_matrix = affine_matrix.copy()
        new_matrix[0:3, 0:3] = R_new
        new_matrix[0:3, -1] = b_new
        return output, new_matrix

def crop_3Dimage(image, center, size, affine_matrix=None):
    """ Crop a 3D image using a bounding box centred at (c0, c1, c2) with specified size (size0, size1, size2) """
    c0, c1, c2 = center
    size0, size1, size2 = size

    S0, S1, S2 = image.shape

    r0, r1, r2 = int(size0 / 2), int(size1 / 2), int(size2 / 2)
    start0, end0 = c0 - r0, c0 + r0
    start1, end1 = c1 - r1, c1 + r1
    start2, end2 = c2 - r2, c2 + r2

    start0_, end0_ = max(start0, 0), min(end0, S0)
    start1_, end1_ = max(start1, 0), min(end1, S1)
    start2_, end2_ = max(start2, 0), min(end2, S2)

    # Crop the image
    crop = image[start0_: end0_, start1_: end1_, start2_: end2_]
    crop = np.pad(crop,
                  ((start0_ - start0, end0 - end0_), (start1_ - start1, end1 - end1_), (start2_ - start2, end2 - end2_)),
                  'constant')

    if affine_matrix is None:
        return crop
    else:
        R, b = affine_matrix[0:3, 0:3], affine_matrix[0:3, -1]
        affine_matrix[0:3, -1] = R.dot(np.array([c0-r0, c1-r1, c2-r2])) + b
        return crop, affine_matrix


''' transform between labelmap and one-hot encoding'''
def onehot2label(seg_onehot):
    "input Cx64x128x128 volume, output 64x128x128, 0 - bg, 1-LV, 2-MYO, 4-RV"
    labelmap = np.argmax(seg_onehot, axis=0)
    tmplabel = labelmap.copy()
    labelmap[tmplabel == 0] = 0
    labelmap[tmplabel == 1] = 1
    labelmap[tmplabel == 2] = 2
    labelmap[tmplabel == 3] = 4
    return labelmap

def label2onehot(labelmap):
    "input 64x128x128 volume, output Cx64x128x128"
    seg_onehot = []
    seg_onehot.append([labelmap == 0])
    seg_onehot.append([labelmap == 1])
    seg_onehot.append([labelmap == 2])
    seg_onehot.append([labelmap == 4])
    return np.concatenate(seg_onehot, axis=0)


''' calculate dice'''
def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))

def np_mean_dice(pred, truth):
    """ Dice mean metric """
    dsc = []
    for k in np.unique(truth)[1:]:
        dsc.append(np_categorical_dice(pred, truth, k))
    return np.mean(dsc)


''' slice volume and plot'''
def vol3view(vol, clim=(0,4), cmap='viridis'):
    " input volume: 64 x 128 x 128"
    plt.style.use('dark_background')
    fig, axs = plt.subplots(nrows=3, ncols=1, frameon=False,
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    fig.set_size_inches([1, 3])
    [ax.set_axis_off() for ax in axs.ravel()]

    view1 = vol[32, :, :]
    view2 = vol[:, 64, :]
    view3 = vol[:, :, 64]

    axs[0].imshow(view1, clim=clim, cmap=cmap)
    axs[1].imshow(view2, clim=clim, cmap=cmap)
    axs[2].imshow(view3, clim=clim, cmap=cmap)

    return fig



def move_3Dimage(image, d):
        """  """
        d0, d1, d2 = d
        S0, S1, S2 = image.shape

        start0, end0 = 0 - d0, S0 - d0
        start1, end1 = 0 - d1, S1 - d1
        start2, end2 = 0 - d2, S2 - d2

        start0_, end0_ = max(start0, 0), min(end0, S0)
        start1_, end1_ = max(start1, 0), min(end1, S1)
        start2_, end2_ = max(start2, 0), min(end2, S2)

        # Crop the image
        crop = image[start0_: end0_, start1_: end1_, start2_: end2_]
        crop = np.pad(crop,
                      ((start0_ - start0, end0 - end0_), (start1_ - start1, end1 - end1_),
                       (start2_ - start2, end2 - end2_)),
                      'constant')

        return crop


def np_categorical_dice_optim(seg, gt, k=1, ds=None):
    if ds is not None:
        _, dh, dw = ds
        seg_1 = move_3Dimage(seg, (0, dh, dw))
        dice_1 = np_categorical_dice(seg_1, gt, k)
        return dice_1

    else:
        d = 5
        max_dice = 0
        best_dh, best_dw = 0, 0
        for dh in range(-d, d):
            for dw in range(-d, d):
                seg_1 = move_3Dimage(seg, (0, dh, dw))
                dice_1 = np_categorical_dice(seg_1, gt, k)
                if dice_1 > max_dice:
                    best_dh, best_dw = dh, dw
                    max_dice = dice_1
        ds = (0, best_dh, best_dw)
        return max_dice, ds


def np_mean_dice_optim(pred, truth, ds):
    """ Dice mean metric """
    dsc = []
    for k in np.unique(truth)[1:]:
        dsc.append(np_categorical_dice_optim(pred, truth, k, ds=ds))
    return np.mean(dsc)