import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pylab as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import FoFCatalogMatching
from dask_ml.model_selection import RandomizedSearchCV
from dask.callbacks import Callback
import dask
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
from collections import defaultdict
import inspect
import pickle  
import bz2
import gzip
import joblib
import random
random.seed(1915)
np.random.seed(1915)

# - some customizations
dask.config.set(scheduler='multiprocessing')
plt.style.use('seaborn-poster')
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# tqdm automatically switches to the text-based
# progress bar if not running in Jupyter
try: # https://github.com/tqdm/tqdm/issues/506
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:  # jupyter
        from tqdm.notebook import tqdm
        # https://github.com/bstriner/keras-tqdm/issues/21
        # this removes the vertical whitespace that remains when tqdm progressbar disappears
        from IPython.core.display import HTML, display
        HTML("""
        <style>
        .p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty {
          padding: 0;
          border: 0;
        }
        </style>
        """)
    if 'terminal' in ipy_str:  # ipython
        from tqdm import tqdm 
except:                        # terminal
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **tqdm_kwargs):
            return iterable

# view larger number of dataframes rows and columns: 
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 25)
pd.set_option('display.min_rows', 15)
pd.set_option('display.expand_frame_repr', True)


# ------------------------
# utilitiy functions
# ------------------------
    
def usedir(mydir,verbose=True):
    # see: https://stackoverflow.com/questions/12468022/python-fileexists-error-when-making-directory
    if not mydir.endswith('/'): mydir += '/' # important
    try:
        if not os.path.exists(os.path.dirname(mydir)):
            os.makedirs(os.path.dirname(mydir)) 
            # sometimes b/w the line above and this line another
            # process may have already made this directory and it
            # leads to [Errno 17]                
            if verbose: logging.info(f' Made directory: {mydir}')
    except OSError as err:
        pass

def drop_trailing_zeros(number):
    """ drop trailing zeros from decimal """
    if number == 0:
        return 0
    else:
        return number.rstrip('0').rstrip('.') if '.' in number else number

def suffixed_format(number, jump=3, precision=2, suffixes=['k', 'M', 'G', 'T', 'P','E','Z','Y','y','z','a','f','p','n','\u03BC','m']):
    """
    < Formatting long numbers as short strings in python >
    Does not work for negative numbers! TODO if needed later.
    precision: applies to floats that don't merely have trailing zeros after the decimal
    suffixes: should be symmetric with even elements

    https://en.wikipedia.org/wiki/Metric_prefix

    Usage:
    
    suffixed_format([[10000.200,43.90008],[67543546,0.005]])

    array([['10k', '43.9'],
       ['67.54M', '5m']], dtype='<U7')

    Note: k is scientifically correct not K
    """
    thresh_exponent = jump*len(suffixes)/2
    suffixes = [''] + suffixes # for cases b/w 10^-jump and 10^jump
    if isinstance(number, (list, np.ndarray)):
        number = np.array(number)
        number_shape = number.shape
        number = number.flatten()
        if any( _n!=0 and (_n<10**(-thresh_exponent) or _n>=10**(thresh_exponent+1)) for _n in number):
            raise ValueError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: At least one number is either too small or too big in the provided data array/list.')
        mth = np.log10(number+(number==0)) // jump
        mth = ( (number<=10**(-jump)) | (number>=10**jump) )*mth.astype(int)
        return np.array([f"{drop_trailing_zeros(f'{_number/10.**(jump*_mth):.{precision}f}')}{suffixes[_mth]}"
                         for _number, _mth in zip(number,mth)]).reshape(number_shape)
    else:
        if number!=0 and (number<10**(-thresh_exponent) or number>=10**(thresh_exponent+1)):
            raise ValueError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The number {number} is either too small or too big.')
        mth = np.log10(number+(number==0)) // jump
        mth = ( (number<=10**(-jump)) | (number>=10**jump) )*mth.astype(int)
        return f"{drop_trailing_zeros(f'{number/10.**(jump*mth):.{precision}f}')}{suffixes[mth]}"
    
# ------------------------
# main class object
# ------------------------

class PhotozBlend(object):

    def __init__(self, truth_df=None, coadd_df=None, zgrid=None):
        self.truth_df = truth_df
        self.coadd_df = coadd_df
        self.zgrid = zgrid
        self.bandwidth = None
        self.bandwidth_tuple = None
        self.bandwidth_array = None
        self.bandwidth_str = None
        self.search_params = None
        self.true_z_hist_smooth = None
        self._plot_fof = DotDict()
        self.refresh_z=True
        self.refresh_pdf=True
        self.refresh_true_z_kernel=True
        self.refresh_pit=True
            
    # --------------------------
    # analysis functions
    # --------------------------

    @staticmethod
    def get_ballpark_estimate_bandwidth(sample, bw_method='scott'):
        " bw_method : 'scott' | 'silverman' "
        kde = stats.gaussian_kde(sample)
        kde.set_bandwidth(bw_method=bw_method)
        factor = kde.covariance_factor()
        bw = factor * sample.std()
        return bw

    # courtesy Sam 
    @staticmethod
    def fastCalcPIT(zgrid,pdf,sz):
        cdf = np.cumsum(pdf)
        idx = np.searchsorted(zgrid,sz,side='left')
        if sz <= zgrid[0]:
            return 0.0
        if sz >= zgrid[-1]:
            return 1.0
        y1,y2 = cdf[idx-1],cdf[idx]
        x1,x2 = zgrid[idx-1],zgrid[idx]
        delx = (sz-x1)*0.5
        if np.isclose(delx,0.0):
            return y1
        else:
            slope = (y2-y1)/(x2-x1)
            finy = y1 + slope*delx
            return finy
        
    @property
    def params(cls):
        " print important parameters (current) "
        
        excluded = ['_plot_fof'] # not interesting
        
        param_df = pd.DataFrame([])
        pd.set_option('display.max_rows', 45)
        pd.set_option('display.min_rows', 45)
        pd.set_option('display.max_colwidth', 60)

        for attr, value in vars(cls).items(): #.keys(): #param_keys:
            if attr in excluded:
                continue
            if isinstance(value, (np.ndarray, list, pd.DataFrame)):
                if isinstance(value, pd.DataFrame):
                    value = value.to_numpy()
                value = np.array2string(np.array(value), max_line_width=30, edgeitems=7, threshold=3)
            row = pd.Series({'value':value},name=attr)
            param_df = param_df.append(row)
        
        return param_df
    
    
    # https://stackoverflow.com/questions/18474791/decreasing-the-size-of-cpickle-objects
    # "The file size of the bzip2 is almost 40x smaller, gzip is 20x smaller.
    # And gzip is pretty close in performance to the raw cPickle"
    def save(self, compression='gzip', fpath='output/pzb.classobj', verbose=True):
        
        fpath = fpath if fpath.startswith(os.getcwd()) else os.getcwd()+'/'+fpath
        if not compression: # raw cPickle
            with open(fpath, 'wb') as f:
                pickle.dump(self, f)
        elif compression == 'bz2': # pipe the pickle through bz2
            with bz2.BZ2File(fpath+'.'+compression, 'w') as f:
                pickle.dump(self, f)
        elif compression == 'gzip': # pipe the pickle through gzip
            with gzip.GzipFile(fpath+'.'+compression, 'w') as f:
                pickle.dump(self, f)
        else:
            raise ValueError('Illegal compression.')
        logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: The class object has been saved in '{fpath+'.'+str(compression or '')}'")
    
    def load(self, compression='gzip', fpath='output/pzb.classobj', verbose=True):

        fpath = fpath if fpath.startswith(os.getcwd()) else os.getcwd()+'/'+fpath
        if not compression: # raw cPickle
            with open(fpath, 'rb') as f:
                classobj = pickle.load(f)
        elif compression == 'bz2': # re-load a zipped pickled object by bz2
            with bz2.BZ2File(fpath+'.'+compression, 'r') as f:
                classobj = pickle.load(f)
        elif compression == 'gzip': # re-load a zipped pickled object by gzip
            with gzip.GzipFile(fpath+'.'+compression, 'r') as f:
                classobj = pickle.load(f)
        else:
            raise ValueError('Illegal compression.')
        logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: The class object has been reloaded from '{fpath+'.'+str(compression or '')}'")
        return classobj
            
            
    # - for the simpler kde_sklearn() version similar to this see: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    def kde_dask(self, data, data_grid, bandwidth='scott', kernel='gaussian', cv=None, n_jobs=None, n_iter=None, leave=False, verbose=True, **kernel_kwargs):
        """Kernel Density Estimation with Dask and Scikit-learn
        data
        data_grid :: point we want to evaluate kde at; bin centers -- it can be finer than the actual histogram we want to fit a model to
        bandwidth -- if not a number --> cross validation cv search is done to find optimum
        kernel
        cv=cv
        n_jobs int or None, optional (default=None)
               Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend
               context. -1 means using all processors. n_jobs = -2 means parallelization in all CPUs
               but 1 (until the previous element from the last, hence -2)
        returns
              data_smooth --> which is a smoothed pdf
        """            
        
        if cv is None:
            cv=3 # 3-Fold

        if n_jobs is None:
            n_jobs=-2 # -2 uses all CPUs but one
        
        if n_iter is None:
            n_iter=10
            
        search_params = {'cv':cv, 'n_iter':n_iter, 'random_state':1905} # only the parameters that affect the final results
            
        if isinstance(bandwidth, str):
            if bandwidth not in ['scott','silverman']:
                raise ValueError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: Illegal bandwidth optimization method.')

        same_bw_str = bandwidth == self.bandwidth_str
        self.bandwidth_str = bandwidth if isinstance(bandwidth, str) else None
        same_bw = np.array_equal(bandwidth,self.bandwidth) # it performs an scalar comparison for an scalar bandwidth
        same_bw_tuple = np.array_equal(bandwidth,self.bandwidth_tuple)
        same_bw_array = np.array_equal(bandwidth,self.bandwidth_array)
        same_serach_params = search_params == self.search_params
        self.search_params = search_params if isinstance(bandwidth, (tuple, list, np.ndarray)) else None

        if self.refresh_true_z_kernel or not (same_bw_str or same_bw or (same_bw_tuple and search_params) or (same_bw_array and search_params)):
            if isinstance(bandwidth, (list, str, tuple, np.ndarray)):
                if isinstance(bandwidth, (str, tuple)):
                    bw_method=bandwidth if isinstance(bandwidth, str) else 'scott'
                    bandwidth_guess = self.get_ballpark_estimate_bandwidth(data, bw_method=bw_method)
                    if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: Estimated the bandwidth of the kernel using the {bw_method} method to be {bandwidth_guess:.3f}.')
                if isinstance(bandwidth, (list, tuple, np.ndarray)):
                    if isinstance(bandwidth, tuple) and len(bandwidth)!=3:
                        raise ValueError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The bandwidth tuple has an illegal length of {len(bandwidth)}. It must be 3.')
                    else:
                        data = data[np.isfinite(data)] # remove possible nans and infs
                        if isinstance(bandwidth, tuple):
                            rtol_low, rtol_high, nbw = bandwidth
                            if rtol_low>=1:
                                raise ValueError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The bandwidth parameter `rtol_low` is {rtol_low}. It must be less than 1.')
                            bw_low = bandwidth_guess*(1-rtol_low)
                            bw_high = bandwidth_guess*(1+rtol_high)
                            bandwidth_candidates = np.linspace(bw_low, bw_high, nbw)
                        elif isinstance(bandwidth, (list, np.ndarray)):
                            bandwidth_candidates = bandwidth
                            bw_low = min(bandwidth_candidates)
                            bw_high = max(bandwidth_candidates)
                            nbw = len(bandwidth_candidates)
                            bandwidth_guess = 0.5*(bw_low+bw_high) # in this case exactly in the middle (cannot potentially be asymmetric like the tuple method)
                        else:
                            raise ValueError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: Illegal bandwidth. It must be a scalar, a tuple of length 3, an array, or a list')
                        # create a model and search the parameter space
                        kde_model = KernelDensity(kernel=kernel, **kernel_kwargs)
                        param_space = {'bandwidth': bandwidth_candidates} # exploring around the guess point
                        if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: Searching for an optimal bandwidth among {nbw} candidates ranging from {min(bandwidth_candidates):.3f} to {max(bandwidth_candidates):.3f}.')                                             
                        search = RandomizedSearchCV(kde_model, param_space, n_jobs=n_jobs,
                                                    refit=True, **search_params) # cv=LeaveOneOut() takes so long
                        with DaskProgressBar(desc='KDE cross-validation', position=0, leave=leave): 
                            with joblib.parallel_backend('threading'):
                                # the `with` line above is not necessary but the following `fit` otherwise hangs for some setups
                                # https://github.com/scikit-learn/scikit-learn/issues/5115
                                search.fit(data[:, None]);
                        bandwidth_best = search.best_params_['bandwidth']
                        # - sklearn: "Due to the high number of test sets (which is the same as the number of samples)
                        #   LeaveOneOut() cross validation (cv) method can be very costly. For large datasets
                        #   one should favor KFold, StratifiedKFold or ShuffleSplit."
                        if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: Optimal bandwidth of the kernel: {bandwidth_best:.3f}')
                        if bandwidth_best>bandwidth_guess:
                            if (bandwidth_best-bandwidth_guess)/(bw_high-bandwidth_guess)>0.9:
                                logging.warning(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The optimal bandwidth={bandwidth_best:.3f} is very close to the higher end of the search interval. Increase your range and try again.')
                        else:
                            if (bandwidth_guess-bandwidth_best)/(bandwidth_guess-bw_low)>0.9:
                                logging.warning(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The optimal bandwidth={bandwidth_best:.3f} is very close to the lower end of the search interval. Increase your range and try again.')
                        if isinstance(bandwidth, tuple):
                            self.bandwidth_tuple = (rtol_low, rtol_high, nbw)
                            self.bandwidth_array = None
                        else:
                            self.bandwidth_array = bandwidth_candidates
                            self.bandwidth_tuple = None
                        self.bandwidth_str = None
                        bandwidth = bandwidth_best
                else:
                    bandwidth = bandwidth_guess
                    self.bandwidth_tuple = None
                    self.bandwidth_array = None
        
            kde_skl = KernelDensity(bandwidth=bandwidth, kernel=kernel, **kernel_kwargs)
            kde_skl.fit(data[:, np.newaxis])
            log_pdf = kde_skl.score_samples(data_grid[:, np.newaxis])
            data_smooth = np.exp(log_pdf)
            self.bandwidth=bandwidth
            self.refresh_true_z_kernel = False
            return data_smooth
        else:
            if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The kde-estimated true-z density funcion remained unchanged since no update was needed.')
            return self.true_z_hist_smooth # returns the saved value from out previous calculation


    def apply_truth_cuts(self, truth_cuts):
        # - cuts should be a list of strings
        filters = ' & '.join(truth_cuts)
        pd.set_option('mode.use_inf_as_na', True) # True means treat None, NaN, INF, -INF as NA (necessery for our pandas query!)
        self.truth_df = self.truth_df.query(filters, engine='python').reset_index(drop=True)
        if hasattr(self,'truth_cuts'):
            if set(self.truth_cuts)!=set(truth_cuts): # i.e. if you are passing new cuts
                self.truth_cuts += truth_cuts # store the entire set of cuts so far applied along with the older ones
        else:
            self.truth_cuts = truth_cuts

    def apply_coadd_cuts(self, coadd_cuts):
        # - cuts should be a list of strings
        filters = ' & '.join(coadd_cuts)
        pd.set_option('mode.use_inf_as_na', True) # True means treat None, NaN, INF, -INF as NA (necessery for our pandas query!)
        self.coadd_df = self.coadd_df.query(filters, engine='python').reset_index(drop=True)
        if hasattr(self,'coadd_cuts'):
            if set(self.coadd_cuts)!=set(truth_cuts): # i.e. if you are passing new cuts
                self.coadd_cuts += coadd_cuts # store the entire set of cuts so far applied along with the older ones
        else:
            self.coadd_cuts = coadd_cuts
            
    def fof_match(self, linking_lengths=1.0, verify=True, plot=True, colorbar='vertical', save_cached=None, load_cached=None, use_latest=None, verbose=True):
        " use_latest is an alias for load_cached "
        
        if load_cached and use_latest:
            raise ValueError('You cannot set `load_cached` and `use_latest` at the same time. `use_latest` is an alias for `load_cached`.')
            
        if use_latest is not None:
            load_cached = use_latest # use_latest is an alias for load_cached just to be consistent with plot_{} methods
        
        self.linking_lengths = linking_lengths

        if save_cached is None:
            save_cached = True if not load_cached else False

        if load_cached is None:
            load_cached = False

        if save_cached or load_cached: # TODO: add custom filename feauture
            self.fof_cache_filename = f'output/fof_results--linking_lengths={self.linking_lengths}.cache'

        if save_cached and load_cached:
            logging.warning(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: You cannot set `save_cached=True` and `load_cached=True` at the same time. One of them is redundant.')

        if load_cached:
            # read in fof_results from cache to avoid doing matching over again
            self.fof_results = pd.read_pickle(self.fof_cache_filename)  
            # - add some necessities that are not in the cache file
            self.truth_mask = self.fof_results['catalog_key'] == 'truth'
            self.coadd_mask = ~self.truth_mask
        else:
            self.linking_lengths = linking_lengths
            # - matching takes some time!
            self.fof_results = FoFCatalogMatching.match(
                catalog_dict={'truth': self.truth_df, 'coadd': self.coadd_df},
                linking_lengths=self.linking_lengths,
                catalog_len_getter=lambda x: len(x['ra']),
            )
            # - FoFCatalogMatching.match returns an astropy table
            #   we convert it to a pandas dataframe
            self.fof_results = self.fof_results.to_pandas()

            # - add the magnitude column
            # https://stackoverflow.com/questions/57137648/getting-data-from-another-dataframe-respect-to-index
            self.fof_results.set_index('row_index', inplace=True, drop=False) # do not drop row_index

            # - first we need to know which rows are from the truth catalog and which are from the coadd
            self.truth_mask = self.fof_results['catalog_key'] == 'truth'
            self.coadd_mask = ~self.truth_mask

            # - now we can add the magnitude columns from the truth and the coadd dataframes, respectively
            self.fof_results.loc[self.truth_mask, 'mag_i_lsst'] = self.truth_df['mag_i_lsst']
            self.fof_results.loc[self.coadd_mask, 'mag_i_lsst'] = self.coadd_df['mag_i_lsst'] 
            self.fof_results.reset_index(drop=True)

            if save_cached: 
                # pickle fof_results to be used later if needed
                usedir(self.fof_cache_filename.rsplit('/',1)[0])
                self.fof_results.to_pickle(self.fof_cache_filename)
                if verbose: 
                    logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: The results of FoF catalog matching is cached in '{self.fof_cache_filename}'\n"+
                       "As long as this file is not deleted, you can load it again by setting load_cached=True.")

        # - count the number of truth and objects per group
        n_groups = self.fof_results['group_id'].max() + 1
        fof_truth = self.fof_results[self.truth_mask]
        fof_coadd = self.fof_results[self.coadd_mask]
        self.n_truth_arr = np.bincount(fof_truth['group_id'], minlength=n_groups)
        self.n_coadd_arr = np.bincount(fof_coadd['group_id'], minlength=n_groups)

        # - now n_truth_arr and n_coadd_arr are the number of truth/coadd objects in each group
        #   let's make a 2d histrogram of (n_truth_arr, n_coadd_arr)
        self.n_max = max(self.n_truth_arr.max(), self.n_coadd_arr.max()) + 1
        self.fof_hist_2d = np.bincount(self.n_coadd_arr * self.n_max + self.n_truth_arr, minlength=self.n_max*self.n_max).reshape(self.n_max, self.n_max)

        if verify:
            # - a quick verification to make sure we didn't mess anything up
            truth_is_matched = (fof_truth.loc[fof_truth['row_index'], 'mag_i_lsst'] == self.truth_df['mag_i_lsst'][fof_truth['row_index']]).all()
            coadd_is_matched = (fof_coadd.loc[fof_coadd['row_index'], 'mag_i_lsst'] == self.coadd_df['mag_i_lsst'][fof_coadd['row_index']]).all()
            if truth_is_matched and coadd_is_matched:
                print(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: FoF results are verified, everything looks good.')
            else:
                raise RuntimeError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The column `mag_i_lsst` has not been inserted in `fof_results` properly.')          

        if plot:
            # - add figure and axis objects
            _fig = plt.figure()
            _ax = _fig.add_subplot(111)
            self.plot_fof(fig=_fig,ax=_ax,colorbar=colorbar,box=False)

    def load_redshifts(self,num_truth=None,num_coadd=None,truth_pick=None,pz_type=None,verbose=True,force_refresh=False):

        if hasattr(self,'num_truth') and hasattr(self,'num_coadd') and hasattr(self,'pz_type'):
            if num_truth==self.num_truth and num_coadd==self.num_coadd and pz_type==self.pz_type:
                if num_truth==1:
                    load=False
                else: # i.e. num_truth>=2
                    if truth_pick is not None:
                        if hasattr(self,'truth_pick'):
                            if truth_pick==self.truth_pick:
                                load=False
                            elif truth_pick in ['bright','faint']:
                                load=True
                            else:
                                raise RuntimeError(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: Illegal value for `truth_pick`: {truth_pick} (choose either 'bright' or 'faint').")
                        else:
                            load=True
                    else:
                        raise RuntimeError(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: Since `num_truth`={num_truth}>1, you should pick a value for `truth_pick` (either 'bright' or 'faint').")
            else:
                load=True
        else:
            load=True

        if load or force_refresh:
            # - warn an unwanted case first?
            if num_coadd > num_truth:
                # it actually needs to raise a value error since we do not handle this atm!
                logging.error(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: Are you sure you want `n_coadd_arr` to be greater than `n_truth_arr`?')

            # - first, let's find out the IDs of the groups that have n_truth_arr to n_coadd_arr truth/coadd match:
            group_mask = np.in1d(self.fof_results['group_id'], np.flatnonzero((self.n_truth_arr == num_truth) & (self.n_coadd_arr == num_coadd)))

            # - and then we can find the row indices in the original truth/coadd catalogs for those match groups
            if num_truth==1:
                self.truth_idx = self.fof_results['row_index'][group_mask & self.truth_mask]
            else:
                if truth_pick=='bright': # take the brightest galaxy for its redshift
                    self.truth_idx = self.fof_results[group_mask & self.truth_mask].groupby('group_id')['mag_i_lsst'].idxmin()
                elif truth_pick=='faint': # take the faintest galaxy for its redshift
                    self.truth_idx = self.fof_results[group_mask & self.truth_mask].groupby('group_id')['mag_i_lsst'].idxmax()
                else:
                    raise RuntimeError(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: Since `num_truth`={num_truth}>1, you should pick a value for `truth_pick` (either 'bright' or 'faint').")

            self.object_idx = self.fof_results['row_index'][group_mask & self.coadd_mask]

            # - find the x and y arrays
            self.true_z = self.truth_df['redshift'][self.truth_idx]
            self.coadd_z = self.coadd_df[pz_type][self.object_idx]

            self.num_truth=num_truth
            self.num_coadd=num_coadd
            self.pz_type=pz_type
            self.truth_pick=truth_pick
            
            self.refresh_z=True
            self.refresh_pdf=True
            self.refresh_true_z_kernel=True
            self.refresh_pit=True
            
            if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: New redshift dataframes have been created.')

        else:
            self.refresh_z=False
            if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The redshift dataframes remained unchanged since no update was needed.')
            # the other three refreshes will remain unchanged
            # they can only be False if later they get called and update their results

    def stack_photoz(self,verbose=True,force_refresh=False):
        if self.refresh_pdf or force_refresh or not hasattr(self,'stacked_pz'):
            self.stacked_pz = (self.coadd_df['photoz_pdf'][self.object_idx]).sum(axis=0)
            top = np.trapz(self.zgrid*self.stacked_pz, x=self.zgrid)
            bottom = np.trapz(self.stacked_pz, x=self.zgrid)
            self.pzmean = top/bottom
            self.refresh_pdf = False
            if verbose: logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: New stacked photoz's have been created.")
        else:
            if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The stacked photoz remained unchanged since no update was needed.')


    def calc_pits(self,leave=False,verbose=True,force_refresh=False):
        if self.refresh_pit or force_refresh or not hasattr(self,'PITS'):
            self.PITS=[]
            # TODO: increase performance - maybe dask pandas?
            for pzpdf, zt in tqdm(zip(self.coadd_df['photoz_pdf'][self.object_idx], self.true_z), total=len(self.true_z), leave=leave, desc='Computing PITs'):
                pit = self.fastCalcPIT(self.zgrid,pzpdf,zt)
                self.PITS.append(pit)
            self.refresh_pit = False
            if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: New PIT values have been created.')
        else:
            if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: The PIT values remained unchanged since no update was needed.')
              

    def apply_function_truth(self, _group, nth=None, sortby=None, query=None):
        truth = self.truth_df.iloc[_group['row_index']]
        if sortby is not None:
            truth = truth.sort_values(**sortby)
        if nth is not None:
            truth = truth.iloc[nth]
        return eval(query)
    
    def get_truth(self, num_truth=2, num_coadd=1, sortby=None, nth=None, query=None):
        group_mask = np.in1d(self.fof_results['group_id'], np.flatnonzero((self.n_truth_arr == num_truth) & (self.n_coadd_arr == num_coadd)))
        res = self.fof_results[group_mask & self.truth_mask].groupby('group_id').apply(self.apply_function_truth, sortby=sortby, nth=nth, query=query)
        return res


    def apply_function_coadd(self, _group, nth=None, sortby=None, query=None):
        coadd = self.coadd_df.iloc[_group['row_index']]
        if sortby is not None:
            coadd = coadd.sort_values(**sortby)
        if nth is not None:
            coadd = coadd.iloc[nth]
        return eval(query)
    
    def get_coadd(self, num_truth=2, num_coadd=1, sortby=None, nth=None, query=None):
        group_mask = np.in1d(self.fof_results['group_id'], np.flatnonzero((self.n_truth_arr == num_truth) & (self.n_coadd_arr == num_coadd)))
        res = self.fof_results[group_mask & self.coadd_mask].groupby('group_id').apply(self.apply_function_coadd, sortby=sortby, nth=nth, query=query)
        return res
    
    # ------------------------
    # plotting functions
    # ------------------------

    def plot_footprints(self, fig=None, figsize=None, ax=None, labels={'truth':'Truth','coadd':'Coadd'}, save_plot=False,
                        verbose=True, plot_dir='output/plots/', save_name='footprints.png', **kwargs):
        """ plot data footprints """
        
        # **kwargs just takes the overflow of unnecessary arguments from the plot_multi() method
        
        if fig is None:
            if ax is not None:
                raise RuntimeError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: You should provide the `fig` argument for the `ax` that you gave.')
        elif figsize is not None:
                logging.warning(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: `figsize` is not needed when the `fig` argument is already given.')
        if fig is None:
            if figsize is None:
                figsize = (9,9)
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)
        
        plt_coadd = ax.hist2d(self.coadd_df['ra'],self.coadd_df['dec'], bins=80, cmap='Blues', alpha=1)
        plt_truth = ax.hist2d(self.truth_df['ra'],self.truth_df['dec'], bins=80, cmap='Oranges', alpha=0.4)
        handles = [matplotlib.patches.Rectangle((0,0),1,1,color=c,ec="k") for c in ['C1','C0']]
        labels = [*labels.values()]
        ax.legend(handles, labels)
        ax.set_xlabel('ra/deg')
        ax.set_ylabel('dec/deg');

        if save_plot:
            usedir(plot_dir)
            if not plot_dir.endswith('/'): plot_dir += '/'
            fpath = plot_dir+save_name
            fpath = fpath.format(**locals())
            fpath = fpath if fpath.startswith(os.getcwd()) else os.getcwd()+'/'+fpath
            fig.savefig(fpath, bbox_inches='tight')
            if verbose: logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: The plot is saved in '{fpath}'")

        
    def plot_fof(self, fig=None, figsize=None, ax=None, colorbar='horizontal', pad='1.3%', annotate=True,
                 box=True, cmap='Blues', colorbar_lim=None, save_plot=False, use_latest=False,
                 plot_dir='output/plots/', save_name='fof.png', verbose=True, **kwargs):
        """ fof matching plot """
        
        if fig is None:
            if ax is not None:
                raise RuntimeError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: You should provide the `fig` argument for the `ax` that you gave.')
        elif figsize is not None:
                logging.warning(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: `figsize` is not needed when the `fig` argument is already given.')
        if fig is None:
            if figsize is None:
                figsize = (9,9)
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)
            
        if use_latest and hasattr(self._plot_fof, 'box'):
            box = self._plot_fof.box
            
        # - assign a logarithmic scale to the histogram values
        self.fof_matrix = np.log10(self.fof_hist_2d+1)
        colorbar_lim = [None,None] if colorbar_lim is None else colorbar_lim
        im = ax.imshow(self.fof_matrix, extent=(-0.5, self.n_max-0.5, -0.5, self.n_max-0.5),
                       origin='lower', cmap=cmap, vmin=colorbar_lim[0], vmax=colorbar_lim[1])
        if colorbar is not None:
            ax_divider = make_axes_locatable(ax)
            # - add an axes above the main axes
            append_to = 'top' if colorbar=='horizontal' else 'right'
            cax = ax_divider.append_axes(append_to, size='7%', pad=pad)
            cbar = fig.colorbar(im, orientation=colorbar, cax=cax);
            # - change tick and label position to top. They default to bottom and overlap the image
            if colorbar=='horizontal':
                cax.xaxis.set_label_position('top')
                cax.xaxis.set_ticks_position('top')
            cax.tick_params(labelsize='x-large')
            cbar.set_label(r'$\log(N_{\rm groups} \, + \, 1)$', labelpad=16, size='xx-large', weight='bold')
        ax.set_xlabel('Number of truth objects');
        ax.set_ylabel('Number of coadd objects');
        # - change the tick frequency to insure we include all the integers
        ax.set_xticks(np.arange(0, self.n_max, 1.0))
        ax.set_yticks(np.arange(0, self.n_max, 1.0))
        # - equalize the scales of x-axis and y-axis
        ax.axis('scaled')
        # - add a rectangle to mark the active 2d bin if needed
        if box:
            if isinstance(box, (tuple, list)): # overrides num_truth and num_coadd
                num_truth, num_coadd = box
            elif hasattr(self,'num_truth') and hasattr(self,'num_coadd'):
                num_truth, num_coadd = self.num_truth, self.num_coadd
            elif kwargs['num_truth'] is not None and kwargs['num_coadd'] is not None:
                num_truth, num_coadd = kwargs['num_truth'], kwargs['num_coadd']
            else:
                logging.warning(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: `num_truth` and `num_coadd` are needed for marking the matrix. Used the default values of 1 and 1.')
                num_truth, num_coadd = 1, 1

            ax.add_patch(patches.Rectangle( (num_truth-0.5, num_coadd-0.5), # (x,y)
                                             1, # width
                                             1, # height
                                             alpha=0.9, facecolor='none', edgecolor='orange',
                                             lw=4, linestyle='solid') )
            self._plot_fof.box = box
            
        if annotate:
            # - loop over the data and create a `Text` for each "pixel"
            #   and change the text's color depending on the data
            for i in range(self.fof_matrix.shape[0]):
                for j in range(self.fof_matrix.shape[1]):
                    # - a list or array of two color specifications. The first is used for values
                    #   below a threshold, the second for those above
                    textcolors = ['black', 'white']
                    # - value in data units according to which the colors from textcolors are
                    #   applied. im.norm(data.max())/2 takes the middle of the colormap as separation
                    threshold = im.norm(self.fof_matrix.max())/2
                    textcolor = textcolors[int(im.norm(self.fof_matrix[i, j]) > threshold)]
                    ax.text(j, i, suffixed_format(self.fof_hist_2d[i,j],precision=0), ha='center', va='center',
                             fontweight='regular', fontsize='x-large', color=textcolor)
        if save_plot:
            usedir(plot_dir)
            if not plot_dir.endswith('/'): plot_dir += '/'
            fpath = plot_dir+save_name
            fpath = fpath.format(**locals())
            fpath = fpath if fpath.startswith(os.getcwd()) else os.getcwd()+'/'+fpath
            fig.savefig(fpath, bbox_inches='tight')
            if verbose: logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: The plot is saved in '{fpath}'")
            
            
    def plot_zz(self, pz_type=None, truth_pick=None, num_truth=None, num_coadd=None, xlim=(0,3), ylim=(0,3), fig=None,
                figsize=None, ax=None, colorbar='horizontal', pad='1.3%', cmap=plt.cm.Spectral_r, annotate=True,
                colorbar_lim=None, force_refresh=False, verbose=True, save_plot=False, use_latest=False,
                plot_dir='output/plots/', save_name='zz-{num_coadd}-{num_truth}{truth_pick}.png'):
        """ z-z plot

            Importrant rule: you should either provide all (pz_type, truth_pick, num_truth, num_coadd) arguments or none of them!
            With one exception of not passing truth_pick if num_truth<2

        """ 

        if force_refresh and use_latest:
            raise ValueError('You cannot set `force_refresh` and `use_latest` at the same time.')
        
        if fig is None:
            if ax is not None:
                raise RuntimeError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: You should provide the `fig` argument for the `ax` that you gave.')
        elif figsize is not None:
                logging.warning(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: `figsize` is not needed when the `fig` argument is already given.')
        if fig is None:
            if figsize is None:
                figsize = (9,9)
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)

        if not use_latest:
            if num_truth is None and num_coadd is None and pz_type is None:
                if hasattr(self,'num_truth') and hasattr(self,'num_coadd') and hasattr(self,'pz_type'):
                    if truth_pick is None and hasattr(self,'truth_pick'):
                        truth_pick=self.truth_pick # use the stored one if necessary
                    self.load_redshifts(num_truth=self.num_truth, num_coadd=self.num_coadd, pz_type=self.pz_type, truth_pick=truth_pick, force_refresh=force_refresh, verbose=verbose)
                else:
                    self.load_redshifts(num_truth=1, num_coadd=1, pz_type='z_mode', truth_pick=truth_pick, force_refresh=force_refresh, verbose=verbose)
            else:
                self.load_redshifts(num_truth=num_truth, num_coadd=num_coadd, pz_type=pz_type, truth_pick=truth_pick, force_refresh=force_refresh, verbose=verbose)

        colorbar_lim = [None,None] if colorbar_lim is None else colorbar_lim
        im = ax.hexbin(self.true_z, self.coadd_z, gridsize=50, bins='log', cmap=cmap, vmin=colorbar_lim[0], vmax=colorbar_lim[1])
        if colorbar is not None:
            ax_divider = make_axes_locatable(ax)
            append_to = 'top' if colorbar=='horizontal' else 'right'
            cax = ax_divider.append_axes(append_to, size='7%', pad=pad)
            cbar = fig.colorbar(im, orientation=colorbar, cax=cax);
            if colorbar=='horizontal':
                cax.xaxis.set_label_position('top')
                cax.xaxis.set_ticks_position('top')
            cax.tick_params(labelsize='x-large')
            cbar.set_label('Count', labelpad=16, size='xx-large', weight='regular')
        ax.set_xlabel('z_truth')
        ax.set_ylabel('z_bpz')
        ax.axis('scaled')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if annotate:
            p0 = max(xlim[0],ylim[0])
            p1 = min(xlim[1],ylim[1])
            ax.plot([p0,p1],[p0,p1], c='w', ls='--', lw=3)
        # - now set the background color to the lowest color
        #   to avoid ugly feautures around the border lines
        ax.set_facecolor(cmap(0.0)) 
        
        if save_plot:
            usedir(plot_dir)
            num_truth = self.num_truth
            num_coadd = self.num_coadd
            
            if not plot_dir.endswith('/'): plot_dir += '/'
            fpath = plot_dir+save_name
            fpath = fpath.format(**locals())
            fpath = fpath if fpath.startswith(os.getcwd()) else os.getcwd()+'/'+fpath
            fig.savefig(fpath, bbox_inches='tight')
            if verbose: logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: The plot is saved in '{fpath}'")
                
            
    def plot_pdf(self, pz_type=None, truth_pick=None, num_truth=None, num_coadd=None, xlim=None, ylim=None, leave=False,
                 verbose=True, fig=None, figsize=None, ax=None, annotate=True, kde_bandwidth='scott', n_iter=10,
                 cv=None, n_jobs=None, force_refresh=False, save_plot=False, plot_dir='output/plots/', use_latest=False,
                 save_name="pdf-{num_coadd}-{num_truth}{truth_pick}.png", **kernel_kwargs):
        """ plot histograms of true and photometric redshifts """

        if force_refresh and use_latest:
            raise ValueError('You cannot set `force_refresh` and `use_latest` at the same time.')
        
        if fig is None:
            if ax is not None:
                raise RuntimeError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: You should provide the `fig` argument for the `ax` that you gave.')
        elif figsize is not None:
                logging.warning(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: `figsize` is not needed when the `fig` argument is already given.')
        if fig is None:
            if figsize is None:
                figsize = (10,8)
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)

        if not use_latest:
            if num_truth is None and num_coadd is None and pz_type is None:
                if hasattr(self,'num_truth') and hasattr(self,'num_coadd') and hasattr(self,'pz_type'):
                    if truth_pick is None and hasattr(self,'truth_pick'):
                        truth_pick=self.truth_pick # use the stored one if necessary
                    self.load_redshifts(num_truth=self.num_truth, num_coadd=self.num_coadd, pz_type=self.pz_type, truth_pick=truth_pick, force_refresh=force_refresh, verbose=verbose)
                else:
                    self.load_redshifts(num_truth=1, num_coadd=1, pz_type='z_mode', force_refresh=force_refresh, verbose=verbose)
            else:
                self.load_redshifts(num_truth=num_truth, num_coadd=num_coadd, pz_type=pz_type, truth_pick=truth_pick, force_refresh=force_refresh, verbose=verbose)

            self.true_z_mean = self.true_z.mean()

            # - calculate stacked photoz pdf (no action if it has already been calculated for this parameters unless `force_refresh` is True)
            self.stack_photoz(force_refresh=force_refresh,verbose=verbose)

            # - p(z) plot
            width = self.zgrid[1]-self.zgrid[0]
            zgrid_edges = np.append(self.zgrid - width/2, [self.zgrid[-1]]) 
            denom = np.sum(np.diff(zgrid_edges) * self.stacked_pz)
            self.stacked_pz_density = self.stacked_pz / denom
            widths = np.ones_like(self.zgrid)*width
            area_tot = (self.stacked_pz_density*widths).sum()

            # - check if the normalization was successful
            np.testing.assert_equal(round(area_tot, ndigits=5), 1.0)
        
        ax.plot(self.zgrid, self.stacked_pz_density, label='Stacked p(z)', alpha=0.6)
        bin_edges_optimized = np.histogram_bin_edges(self.true_z, bins='auto')
        ax.hist(self.true_z, bins=bin_edges_optimized, histtype='step', lw=1.5, label='True z', alpha=0.3, density=True)
        if self.refresh_true_z_kernel or self.true_z_hist_smooth is None or not use_latest:
            self.true_z_hist_smooth = self.kde_dask(self.true_z, self.zgrid,
                                                    bandwidth=kde_bandwidth, kernel='gaussian', verbose=verbose,
                                                    cv=cv, n_jobs=n_jobs, n_iter=n_iter, leave=leave, **kernel_kwargs) 
        ax.plot(self.zgrid, self.true_z_hist_smooth, label='Smoothed True z', linewidth=3, alpha=0.6, ls='-', color='C1')
        ax.axvline(self.pzmean, ls='--', color='C0', lw=1.5, alpha=0.9)
        ax.axvline(self.true_z_mean, ls='--', color='C1', lw=1.5, alpha=0.9)
        ax.set_xlabel('z')
        ax.set_ylabel('Normalized count')
        ax.ticklabel_format(style="sci", scilimits=(0,0))
        handles, labels = ax.get_legend_handles_labels()
        order = [2,1,0]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order]);
        
        if save_plot:
            usedir(plot_dir)
            num_truth = self.num_truth
            num_coadd = self.num_coadd
            if hasattr(self, 'truth_pick'):
                truth_pick = '-'+self.truth_pick if self.truth_pick else ''
            else:
                truth_pick = ''
            if not plot_dir.endswith('/'): plot_dir += '/'
            fpath = plot_dir+save_name
            fpath = fpath.format(**locals())
            fpath = fpath if fpath.startswith(os.getcwd()) else os.getcwd()+'/'+fpath
            fig.savefig(fpath, bbox_inches='tight')
            if verbose: logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: The plot is saved in '{fpath}'")

    def plot_pit(self, pz_type=None, truth_pick=None, num_truth=None, num_coadd=None, xlim=None, ylim=None, leave=False, verbose=True,
                 fig=None, figsize=None, ax=None, annotate=True, kde_bandwidth='scott', k_splits=None, cv=None, n_jobs=None, use_latest=False,
                 force_refresh=False, save_plot=False, plot_dir='output/plots/', save_name='pit-{num_truth}-{num_coadd}{truth_pick}.png', **kernel_kwargs):
        """ plot a histogram of the PIT value """

        if force_refresh and use_latest:
            raise ValueError('You cannot set `force_refresh` and `use_latest` at the same time.')
        
        if fig is None:
            if ax is not None:
                raise RuntimeError(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: You should provide the `fig` argument for the `ax` that you gave.')
        elif figsize is not None:
                logging.warning(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: `figsize` is not needed when the `fig` argument is already given.')
        if fig is None:
            if figsize is None:
                figsize = (10,8)
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)

        if not use_latest:
            if num_truth is None and num_coadd is None and pz_type is None:
                if hasattr(self,'num_truth') and hasattr(self,'num_coadd') and hasattr(self,'pz_type'):
                    if truth_pick is None and hasattr(self,'truth_pick'):
                        truth_pick=self.truth_pick # use the stored one if necessary
                    self.load_redshifts(num_truth=self.num_truth, num_coadd=self.num_coadd, pz_type=self.pz_type, truth_pick=truth_pick, force_refresh=force_refresh, verbose=verbose)
                else:
                    self.load_redshifts(num_truth=1, num_coadd=1, pz_type='z_mode', force_refresh=force_refresh, verbose=verbose)
            else:
                self.load_redshifts(num_truth=num_truth, num_coadd=num_coadd, pz_type=pz_type, truth_pick=truth_pick, force_refresh=force_refresh, verbose=verbose)

            # - calculate PIT values
            self.calc_pits(leave=leave, force_refresh=force_refresh, verbose=verbose)
        
        # - plot PIT histogram for the entire sample, no tomographic binning 
        bin_edges_optimized = np.histogram_bin_edges(self.PITS, bins='auto')
        ax.hist(self.PITS, bins=bin_edges_optimized, histtype='step', lw=2, density=True, alpha=0.85);
        ax.set_xlabel('PIT')
        ax.set_ylabel('Normalized count')
        ax.axhline(1.0, ls='--', color='grey', lw=1.5, alpha=0.9);
        ax.ticklabel_format(style='sci', scilimits=(0,0))

        if save_plot:
            usedir(plot_dir)
            num_truth = self.num_truth
            num_coadd = self.num_coadd
            truth_pick = '-'+self.truth_pick if self.truth_pick else ''
            if not plot_dir.endswith('/'): plot_dir += '/'
            fpath = plot_dir+save_name
            fpath = fpath.format(**locals())
            fpath = fpath if fpath.startswith(os.getcwd()) else os.getcwd()+'/'+fpath
            fig.savefig(fpath, bbox_inches='tight')
            if verbose: logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: The plot is saved in '{fpath}'")

    def plot_multi(self, names=['fof','zz','pdf','pit'], num_truth=None, num_coadd=None, truth_pick=None, pz_type=None,
                   figsize=(15,15), nrows=2, ncols=2, height_ratios=[1.53, 1], suptitle=None, force_refresh=False, use_latest=False,
                   save_plot=False, verbose=False, plot_dir='output/plots/', save_name='multi-{num_truth}-{num_coadd}{truth_pick}.png', **gs_kwargs):
        
        if force_refresh and use_latest:
            raise ValueError('You cannot set `force_refresh` and `use_latest` at the same time.')
        
        # - add a figure object for the multiplot
        fig = plt.figure(figsize=figsize)

        # - some aesthetic adjustments using GridSpec
        grid = matplotlib.gridspec.GridSpec(nrows, ncols, height_ratios=height_ratios, **gs_kwargs)
                
        for s, name in enumerate(names):
            ax = fig.add_subplot(grid[s])
            method_to_call = getattr(self, f'plot_{name}')
            method_to_call(fig=fig, ax=ax, num_truth=None, num_coadd=None, truth_pick=None,
                           pz_type=None, force_refresh=force_refresh, verbose=verbose, use_latest=use_latest)
        
        if hasattr(self, 'num_truth'):  
            if suptitle is None: # the loop above has already supplied (self.num_truth, self.num_coadd, self.truth_pick)
                suptitle = f"n_truth={self.num_truth}, n_coadd={self.num_coadd}\n{f' (the true redshift of the {self.truth_pick} galaxy is picked for the unrecognized blend)' if self.num_truth>1 else ''}"
            multiline = True if self.num_truth>1 else False
        else:
            multiline = False
        
        if suptitle:
            fig.suptitle(suptitle, size=16, fontweight='bold', va='top', y=1.02 if multiline else 1)
        fig.tight_layout(pad=3.0)

        if save_plot:
            usedir(plot_dir)
            num_truth = self.num_truth if hasattr(self, 'num_truth') else 'x' 
            num_coadd = self.num_coadd if hasattr(self, 'num_truth') else 'x'
            if hasattr(self, 'truth_pick'):
                truth_pick = '-'+self.truth_pick if self.truth_pick else ''
            else:
                truth_pick = ''
            if not plot_dir.endswith('/'): plot_dir += '/'
            fpath = plot_dir+save_name
            fpath = fpath.format(**locals())
            fpath = fpath if fpath.startswith(os.getcwd()) else os.getcwd()+'/'+fpath
            fig.savefig(fpath, bbox_inches='tight')
            if verbose: logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: The plot is saved in '{fpath}'")
                
# http://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class DotDict(defaultdict):    
    def __init__(self): 
        super().__init__()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
            
    def __setattr__(self, key, value):
        self[key] = value
        
        
# modified from https://github.com/tqdm/tqdm/issues/278
# Non-tqdm alternative: from dask.diagnostics import ProgressBar as DaskProgressbar
class DaskProgressBar(Callback):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _start_state(self, dsk, state):
        self._tqdm = tqdm(total=sum(len(state[k]) for k in ['ready', 'waiting', 'running', 'finished']), **self.kwargs)

    def _posttask(self, key, result, dsk, state, worker_id):
        self._tqdm.update(1)

    def _finish(self, dsk, state, errored):
        self._tqdm.close()
        pass
