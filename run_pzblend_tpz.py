import numpy as np
import pandas as pd
import GCRCatalogs
from GCR import GCRQuery
from pzblend import PhotozBlend

mlz = "/global/cfs/cdirs/lsst/groups/PZ/PZBLEND/SPRINTWEEK/output/results/Run2.2i_dr6_100k.11.mlz"
pdfs = "/global/cfs/cdirs/lsst/groups/PZ/PZBLEND/SPRINTWEEK/output/results/Run2.2i_dr6_100k.11.P.npy"
validation = "/global/cfs/cdirs/lsst/groups/PZ/PZBLEND/Run2.2i_dr6/Run2.2i_dr6_dered_test.txt"

mlz_np = np.genfromtxt(mlz)
# the last entry in the pdfs is the zs
pdfs_np = np.load(pdfs)
zs = pdfs_np[-1]
pdfs_np = pdfs_np[:-1]
valid_np = np.genfromtxt(validation)

bands = ['u','g','r','i','z','y']
errs = ['e{}'.format(i) for i in bands]
colors = ['u-g','g-r','r-i','i-z','z-y']
valid_names = bands + colors + errs + ['ra','dec','objId']

mlz_names = ["ztrue",'z_mode', "zmean","zConf0",'zConf1','err0','err1']
pdf_names = "photoz_pdf"

mlz_df = pd.DataFrame(mlz_np, columns=mlz_names)
valid_df = pd.DataFrame(valid_np, columns=valid_names)
valid_df['photoz_pdf'] = list(pdfs_np)

coadd_df = pd.concat([valid_df, mlz_df], axis=1)
coadd_df['mag_i_lsst'] = coadd_df['i']

truth_cat = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_small')
truth_data = truth_cat.get_quantities(['ra', 'dec', 'galaxy_id','halo_id', 'redshift','mag_i', 'mag_i_lsst',
                                       'mag_g', 'mag_r'],filters=['mag_i_lsst < 28.','dec < -38.5'])
truth_df = pd.DataFrame(truth_data)

pzb = PhotozBlend(truth_df, coadd_df, zs)
pzb.fof_match(verify=True, plot=False, load_cached=True) #save_cached=True
