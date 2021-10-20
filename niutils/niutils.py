#!/usr/bin/env python3

import os

import nibabel as nib
import numpy as np

from copy import deepcopy

SUB_LIST = ['001', '002', '003', '004', '007', '008', '009']
LAST_SES = 10  # 10

LAST_SES += 1

SET_DPI = 300
FIGSIZE = (18, 10)

COLOURS = ['#1f77b4ff', '#2ca02cff', '#d62728ff', '#ff7f0eff', '#ff33ccff']


#########
# Utils #
#########
def check_ext(fname, ext='.nii.gz'):
    """
    Check the extension of input. It's possible to add it.
    """
    if fname.endswith(ext):
        fname = fname[:-len(ext)]
    return f'{fname}{ext}'


def load_nifti_get_mask(fname, dim=4):
    """
    Load a nifti file and returns its data, its image, and a 3d mask.
    """
    img = nib.load(check_ext(fname))
    data = img.get_fdata()
    if len(data.shape) > dim:
        for ax in range(dim, len(data.shape)):
            data = np.delete(data, np.s_[1:], axis=ax)
    data = np.squeeze(data)
    if len(data.shape) >= 4:
        mask = np.squeeze(np.any(data, axis=-1))
    else:
        mask = (data < 0) + (data > 0)
    return data, mask, img


def compute_rank(data, islast=True):
    """
    Compute the ranks in the last axis of a matrix.

    It assumes that the "target" data is appended as last element in the axis.
    This is useful to compare e.g. a bunch of surrogates to real data.
    """
    reord = np.argsort(data, axis=-1)
    if islast:
        rank = reord.argmax(axis=-1)
    else:
        rank = reord.argmin(axis=-1)
    return rank/(data.shape[-1]-1)*100


def export_nifti(data, img, fname, overwrite=True):
    """
    Export a nifti file.
    """

    tmp_dim = np.ones(img.header['dim'].shape, dtype=np.int16)
    tmp_dim[0] = len(data.shape)
    tmp_dim[1:len(data.shape)+1] = data.shape

    tmp_header = deepcopy(img.header)

    if (tmp_dim != img.header['dim']).any():
        if not overwrite:
            print(f'!!! Warning: exporting data with shape {data.shape} with '
                  f'a header declaring dimensions: {img.header["dim"]}. ')
        else:
            print('!!! Warning: data shape and header do not match. '
                  'Overwriting header dim and pixdim')
            tmp_pixdim = tmp_header['pixdim']
            tmp_pixdim[len(data.shape)+1:img.header['dim'][0]+1] = 1
            tmp_header['dim'] = tmp_dim
            # tmp_header['pixdim'] = tmp_pixdim  # This line is not necessary!

    out_img = nib.Nifti1Image(data, img.affine, tmp_header)
    out_img.to_filename(check_ext(fname))


def vol_to_mat(data):
    """
    Reshape nD into 2D
    """
    return data.reshape(((-1,) + data.shape[3:]), order='F')


def mat_to_vol(data, shape=None, asdata=None):
    """
    Reshape 2D into nD using either shape or data shape)
    """
    if asdata is not None:
        if shape is not None:
            print('Both shape and asdata were defined. '
                  f'Overwriting shape {shape} with asdata {asdata.shape}')
        shape = asdata.shape
    elif shape == '':
        raise ValueError('Both shape and asdata are empty. '
                         'Must specify at least one')

    return data.reshape(shape, order='F')


def apply_mask(data, mask):
    """
    Reduce shape and size of data based on mask
    """
    if data.shape[:len(mask.shape)] != mask.shape:
        raise ValueError(f'Cannot mask data with shape {data.shape} using mask '
                         f'with shape {mask.shape}')
    if (len(data.shape)-len(mask.shape)) > 1:
        print('Warning: returning volume with '
              f'{len(data.shape)-len(mask.shape)+1} dimensions.')
    else:
        print(f'Returning {len(data.shape)-len(mask.shape)+1}D array.')

    return data[mask != 0]


def unmask(data, mask, shape=None, asdata=None):
    """
    Unmask 1D or 2D into an nD based on shape or asdata
    """
    if asdata is not None:
        if shape is not None:
            print('Both shape and asdata were defined. '
                  f'Overwriting shape {shape} with asdata {asdata.shape}')
        shape = asdata.shape
    elif shape == '':
        raise ValueError('Both shape and asdata are empty. '
                         'Must specify at least one')

    if data.shape[0] != mask.sum():
        raise ValueError('Cannot unmask data with first dimension '
                         f'{data.shape[0]} using mask with shape '
                         f'{mask.shape} ({np.prod(mask.shape)} entries)')
    if shape[:len(mask.shape)] != mask.shape:
        raise ValueError(f'Cannot unmask data into shape {shape} using mask '
                         f'with shape {mask.shape}')

    out = np.zeros(shape)
    out[mask != 0] = data
    return out


def som(data_mkd, n_comp, max_iter=1000, tolerance=-1, init_mode='pca'):
    """
    init_mode: pca, rand
    """
    from sklearn.preprocessing import scale

    if init_mode == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(data_mkd)
        t = pca.components_[:n_comp]
    elif init_mode == 'rand':
        rng = np.random.default_rng(42)
        t = rng.random((n_comp, data_mkd.shape[-1]))
    else:
        raise Exception(f'Init mode {init_mode} not implemented yet. '
                        'Possible modes are: pca, rand')

    data_std = scale(data_mkd, axis=1)
    t_std = scale(t, axis=1)
    tmp_old = np.empty((data_std.shape[0], t_std.shape[0]))

    for i in range(0, max_iter):
        tmp = np.matmul(data_std, t_std.T)
        clus = np.argmax(tmp, axis=1)

        for j in range(0, n_comp):
            t_std[j, :] = data_std[clus == j, :].mean(axis=0)
        t_std = scale(t_std, axis=1)

        print(f'Round: {i}, Distance: {np.linalg.norm(tmp-tmp_old):.6f}')

        if tolerance >= 0 and np.linalg.norm(tmp-tmp_old) <= tolerance:
            break

        tmp_old = np.copy(tmp)

    return clus+1

#############
# Workflows #
#############


def variance_weighted_average(fname,
                              fdir='',
                              exname='',
                              sub_list=SUB_LIST,
                              last_ses=LAST_SES):
    """
    Compute the variance weighted average of a multi-session study.

    It's supposed that:
    - all files are in the same folder
    - `fname` contains placeholders `{sub}` and `{ses}`
    """
    # Prepare dictionaries
    mask = dict.fromkeys(sub_list)
    data = dict.fromkeys(sub_list)
    data['avg'] = dict.fromkeys(sub_list)
    data['var'] = dict.fromkeys(sub_list)

    if fdir:
        fname = os.path.join(fdir, fname)
    elif os.path.split(fname)[0]:
        fdir = os.path.split(fname)[0]

    # Load niftis of all subjects
    for sub in sub_list:
        data[sub] = {}
        mask[sub] = {}
        for ses in range(1, last_ses):
            # Load data
            fname = fname.format(sub=sub, ses=f'{ses:02g}')
            data[sub][ses], mask[sub][ses], img = load_nifti_get_mask(fname, dim=3)

        # Stack in 4d (axis 3) and mask data (invert nimg mask for masked array)
        mask[sub]['stack'] = np.stack(mask[sub].values(), axis=3)
        data[sub]['stack'] = np.ma.array(np.stack(data[sub].values(), axis=3),
                                         mask=abs(mask[sub]['stack']-1))

        # Compute average & variance of masked voxels across d4
        data['avg'][sub] = data[sub]['stack'].mean(axis=3)
        data['var'][sub] = ((data[sub]['stack'] -
                             data['avg'][sub][:, :, :, np.newaxis])**2).mean(axis=3)

    # Stack subjects in 4d
    for val in ['avg', 'var']:
        data[val]['all'] = np.stack(data[val].values(), axis=3)

    # Invert variance & set infinites to zero (if any)
    invvar = 1 / data['var']['all']
    invvar[np.isinf(invvar)] = 0

    # Mask group average using invvar
    data['avg']['all'] = np.ma.array(data['avg']['all'], mask=[invvar == 0])

    # Finally, compute variance weighted average & fill masked entries with 0
    wavg = np.ma.average(data['avg']['all'], weights=invvar, axis=3).filled(0)

    # Export
    if not exname and fdir:
        exname = os.path.split(fdir)[-1]
    if not exname or exname == '.':
        exname = 'wavg'
    else:
        exname = f'wavg_{exname}'
    export_nifti(wavg.astype(float), img, exname)


def variance_weighted_average_volume(data, n_subs=len(SUB_LIST)):
    """
    Compute the variance weighted average of a multi-session study in a volume.

    It's supposed that:
    - data is a 4D volume in which the 4th D is sub 1 ses 1, sub 1 ses 2, ..., sub n ses m
    - OR, data is a 5D volume in which the 4th D is sessions and the 5th D is subjects.
    - The number of sessions is always the same across subjects.

    Might return slightly different results from function above because mask.
    """
    subs = np.split(data, n_subs, axis=-1)
    sub_avg = np.empty([data.shape[0], data.shape[1], data.shape[2], n_subs])
    sub_var = np.empty([data.shape[0], data.shape[1], data.shape[2], n_subs])
    # Load niftis of all subjects
    for n, sub in enumerate(subs):
        # Compute average & variance of voxels
        sub_avg[:, :, :, n] = sub.mean(axis=3)
        sub_var[:, :, :, n] = ((sub - sub_avg[:, :, :, n, np.newaxis])**2).mean(axis=3)

    # Invert variance & set infinites to zero (if any)
    invvar = 1 / sub_var
    invvar[np.isinf(invvar)] = 0

    # Mask group average using invvar
    group = np.ma.array(sub_avg, mask=[invvar == 0])

    # Finally, compute variance weighted average & fill masked entries with 0
    wavg = np.ma.average(group, weights=invvar, axis=3).filled(0)

    return wavg


def compute_metric(data, atlas, mask, metric='avg', invert=False):
    """
    Compute a metric (e.g. average) in the parcels of an atlas.

    The metric is computed in the last axis of `data`, and it assumes that the
    "target" map is the last element in the axis.

    It then returns the metric in the "target" map and its rank equivalent.
    """
    print(f'Compute metric {metric} in atlas')
    atlas = atlas*mask
    unique = np.unique(atlas)
    unique = unique[unique > 0]
    print(f'Labels: {unique}, len: {len(unique)}, surr: {data.shape[-1]}')
    # Initialise dataframe and dictionary for series
    parcels = np.empty([len(unique), data.shape[-1]])

    # Compute averages
    for m, label in enumerate(unique):
        print(f'Metric: {metric}, Label: {label} ({m})')
        if metric == 'avg':
            parcels[m, :] = data[atlas == label].mean(axis=0)
        elif metric == 'iqr':
            dist = data[atlas == label]
            parcels[m, :] = (np.percentile(dist, 75, axis=0) -
                             np.percentile(dist, 25, axis=0))
        elif metric == 'var':
            dist = data[atlas == label]
            parcels[m, :] = data[atlas == label].var(axis=0)

    rank = compute_rank(parcels)
    if invert:
        print(f'Invert {metric} rank')
        rank = 100 - rank

    rank_map = atlas.copy()
    orig_metric = atlas.copy()

    print('Recompose atlas with rank')
    for m, label in enumerate(unique):
        rank_map[atlas == label] = rank[m]

    print(f'Recompose atlas with computed metric ({metric})')
    for m, label in enumerate(unique):
        orig_metric[atlas == label] = parcels[m, -1]

    return rank_map, orig_metric


def compute_som(fname, n_comp, outname='', max_iter=1000, tolerance=-1, init_mode='pca'):
    """
    init_mode: pca, rand
    """
    data, mask, img = load_nifti_get_mask(check_ext(fname))
    data_mkd = apply_mask(data, mask)

    clus = som(data_mkd, n_comp, max_iter, tolerance, init_mode)

    clus_vol = unmask(clus, mask, asdata=mask)

    if outname == '':
        outname = f'som_{init_mode}_{n_comp}'
    export_nifti(clus_vol, img, outname)
