import pandas as pd
from IPython.display import display, Markdown

class Config:
    def __init__(self):
        pass 

    def pdconfig(self,nrows=None,cols_width=None,precision=3):
        pd.set_option("display.max_rows", nrows)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", cols_width)
        pd.set_option("display.precision", precision)
        pd.set_option("display.expand_frame_repr", False)
     

        display(Markdown("Pandas config display options set."))

