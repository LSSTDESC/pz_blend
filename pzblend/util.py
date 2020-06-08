import os
import logging
import numpy as np
import re
import inspect

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
            if verbose: logging.info(f'{inspect.stack()[1].function}:{inspect.stack()[0].function}: Made directory: {mydir}')
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

# courtesy of https://gist.github.com/bgusach/a967e0587d6e01e889fd1d776c5f3729
def multireplace(string, replacements, ignore_case=False):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
    # If case insensitive, we need to normalize the old string so that later a replacement
    # can be found. For instance with {"HEY": "lol"} we should match and find a replacement for "hey",
    # "HEY", "hEy", etc.
    if ignore_case:
        def normalize_old(s):
            return s.lower()

        re_mode = re.IGNORECASE

    else:
        def normalize_old(s):
            return s

        re_mode = 0

    replacements = {normalize_old(key): val for key, val in replacements.items()}
    
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)
    
    # Create a big OR regex that matches any of the substrings to replace
    pattern = re.compile("|".join(rep_escaped), re_mode)
    
    # For each match, look up the new string in the replacements, being the key the normalized old string
    return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)

def translate_easy_string(easy_string, keys=None, prefix='dataframe' , verbose=True, dfdesc='dataframe'):
    translations = [f'{prefix}.{key}' for key in keys]
    translator_dict = dict(zip(keys, translations))
    if isinstance(easy_string, list):
        easy_string_translated = [multireplace(es, translator_dict, ignore_case=False) for es in easy_string]
    else:
        easy_string_translated = multireplace(easy_string, translator_dict, ignore_case=False)
        easy_string = f"\'{easy_string}\'"
    if verbose:
        logging.info(f"{inspect.stack()[1].function}:{inspect.stack()[0].function}: Evaluating {easy_string} in the {dfdesc}") # just to double check in case of a surprise
    return easy_string_translated

def qmatch(query, bank, out='match'):
    matches = [match for match in bank if match in query]
    non_matches = [match for match in bank if match not in query]
    if out=='match':
        return matches
    elif out=='nonmatch':
        return non_matches
    elif out=='all':
        return matches, non_matches
    else:
        raise ValueError('Illegal value for `out`.')

def ordinal(number):
    return f"{number}{'tsnrhtdd'[(np.floor(number/10)%10!=1)*(number%10<4)*number%10::4]}"

