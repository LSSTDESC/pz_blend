from pandas.testing import assert_frame_equal

class testing:
    def all_equal(*dfs, verbose=True, **pd_kwargs):
        '''
        Checks if all the dataframes given as arguments are equal
        
        raises:
            AssertionError if the dataframes are not equal
        
        Example:
        > all_equal(df1, df2, df3, verbose=True, check_dtype=False)
        The dataframes are all equal.
        '''
        checkmark = '\x1b[0;32m'+u'\N{check mark}'+'\x1b[1;32m'
        df_ref = dfs[0]
        for df in dfs[1:]:
            assert_frame_equal(df_ref, df, **pd_kwargs)
        if verbose:
            print(checkmark+' The dataframes are all equal.')
            # see https://stackoverflow.com/questions/16816013/is-it-possible-to-print-using-different-colors-in-ipythons-notebook
