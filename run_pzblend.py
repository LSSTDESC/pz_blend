import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dustmaps
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
from dustmaps.config import config
from pzblend import PhotozBlend
sys.path.insert(0,"/global/cfs/cdirs/lsst/groups/PZ/PhotoZDC2/run2.2i_dr6_test/gcr-catalogs/lib/python3.7/site-packages/GCRCatalogs-0.18.1-py3.7.egg")
import GCRCatalogs
from GCR import GCRQuery

object_cat = GCRCatalogs.load_catalog('dc2_object_run2.2i_dr6a_with_photoz')

tract_ids = [2731, 2904, 2906, 3081, 3082, 3084, 3262, 3263, 3265, 3448, 3450, 3831, 3832, 3834, 4029, 4030, 4031, 2905, 3083, 3264, 3449, 3833]

basic_cuts = [
    GCRQuery('extendedness > 0'),     # Extended objects
    GCRQuery((np.isfinite, 'mag_i')), # Select objects that have i-band magnitudes
    GCRQuery('clean'), # The source has no flagged pixels (interpolated, saturated, edge, clipped...) 
                       # and was not skipped by the deblender
    GCRQuery('xy_flag == 0'), # Bad centroiding
    GCRQuery('snr_i_cModel >= 10'),
    GCRQuery('detect_isPrimary'), # (from this and below) basic flag cuts 
    ~GCRQuery('deblend_skipped'),
    ~GCRQuery('base_PixelFlags_flag_edge'),
    ~GCRQuery('base_PixelFlags_flag_interpolatedCenter'),
    ~GCRQuery('base_PixelFlags_flag_saturatedCenter'),
    ~GCRQuery('base_PixelFlags_flag_crCenter'),
    ~GCRQuery('base_PixelFlags_flag_bad'),
    ~GCRQuery('base_PixelFlags_flag_suspectCenter'),
    ~GCRQuery('base_PixelFlags_flag_clipped')
]

mag_filters = [
    (np.isfinite, 'mag_i'),
    'mag_i < 25.',
]

object_df_list = []
for i in tract_ids:
    object_data = object_cat.get_quantities(['ra','dec','objectId', 'mag_i_cModel', 'magerr_i_cModel',
                                      'mag_r_cModel', 'magerr_r_cModel',
                                             'mag_g_cModel', 'magerr_g_cModel','z_mode','photoz_pdf'],
                                      filters=basic_cuts+mag_filters, native_filters=['tract == {}'.format(i)])
    object_df_list.append(pd.DataFrame(object_data))
coadd_df = pd.concat(object_df_list)

# deredden the cModel magnitudes
band_a_ebv = np.array([4.81,3.64,2.70,2.06,1.58,1.31])
coords = c = SkyCoord(coadd_df['ra'], coadd_df['dec'], unit = 'deg',frame='fk5')
sfd = SFDQuery()
ebvvec = sfd(coords)
coadd_df['ebv'] = ebvvec
coadd_df['mag_i_lsst'] = coadd_df['mag_i_cModel'] - coadd_df['ebv']*band_a_ebv[3]

truth_cat = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_small')

truth_data = truth_cat.get_quantities(['ra', 'dec', 'galaxy_id','halo_id', 'redshift','mag_i', 'mag_i_lsst',
                                       'mag_g', 'mag_r'],filters=['mag_i_lsst < 28.','dec < -38.5'])
truth_df = pd.DataFrame(truth_data)

zgrid_filename = 'data/zgrid.npy'
zgrid = np.load(zgrid_filename)

pzb = PhotozBlend(truth_df, coadd_df, zgrid)
pzb.fof_match(verify=True, plot=False, save_cached=True) #load_cached=True

