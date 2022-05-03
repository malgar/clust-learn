"""Data processing"""
# Author: Miguel Alvarez


class DataProcessing(object):

    def __init__(self,
                 df,
                 num_vars=None,
                 cat_vars=None):

        self.df = df
        self.num_vars = num_vars
        self.cat_vars = cat_vars

