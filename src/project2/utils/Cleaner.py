import pandas as pd 
import pycountry
import re 
from IPython.display import display, HTML
from project2.data.Saver import Saver


class Cleaner:
    """Data cleaning utilities for country data and date formatting."""

    def __init__(self, name=None, df=None, saver=None):
        self.name = name
        self.df = df
        self.saver = saver if saver is not None else Saver()

    def attach_data(self, name, df):
        """Attach dataset name and DataFrame to the cleaner instance."""
        self.name = name
        self.df = df
        if self.saver is None:
            self.saver = Saver()

    def format_year(self, list_clean):
        """Extract and format year columns from mixed date formats."""
        columns = self.df.columns.tolist()
        
        for col in columns:
            if col in list_clean:
                # Extract 4-digit years from strings
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.extract(r'(\d{4})', expand=False)
                    .fillna("")
                    .astype("string")
                ) 
                self.df[col] = self.df[col].astype(object)
                
                print(f"After cleaning: {self.name} - {list(self.df.columns)}")

        # Update and save
        self.attach_data(self.name, self.df)
        self.saver.get_file(self.name, self.df)
        self.saver.save_process_files(self.name, self.df, 
                                      path="processed", filetype='csv')

    def year_extractor(self, val):
        """Extract 4-digit year from a value."""
        if pd.isna(val):
            return ''
        
        if isinstance(val, str):
            match = re.search(r"\d{4}", val)
            return match.group(0) if match else val
        
        return str(val)

    def fk_country(self, name, df, return_code=True, disp=True):
        """
        Identify invalid country codes using pycountry library.
        
        Parameters
        ----------
        name : str
            Dataset name
        df : pd.DataFrame
            DataFrame containing country codes
        return_code : bool
            Whether to return list of invalid codes
        disp : bool
            Whether to display results
            
        Returns
        -------
        list or None
            List of invalid country codes if return_code=True
        """
        # Reference country codes from pycountry
        ref_country_code = [country.alpha_3.upper() for country in pycountry.countries]
        
        # Find country code column
        col_name = None
        for col in df.columns:
            if col in ['Country Code', 'CountryCode']:
                col_name = col
                break
        
        if col_name is None:
            print(f'{name} - Country Code column not found')
            return None
        
        # Identify invalid country codes
        country_code_data = df[col_name]
        invalid_country_code = [
            code.upper() for code in country_code_data
            if isinstance(code, str) and code.upper() not in ref_country_code
        ]
        
        print(f"For {name} - {len(invalid_country_code)} invalid country codes identified")
        
        # Display invalid entries
        if disp or return_code:
            df_invalid = df[df[col_name].isin(invalid_country_code)]
            columns_show = [col_name, 'Short Name'] if 'Short Name' in df.columns else [col_name]
            df_display = df_invalid[columns_show].reset_index(drop=True)
            
            if disp:
                display(HTML(df_display.to_html(max_rows=None, max_cols=None)))
        
        return invalid_country_code if return_code else None
    

    def removefkcountry(self, name, df):
        """
        Remove rows with invalid country codes from dataset.
        
        Parameters
        ----------
        name : str
            Dataset name
        df : pd.DataFrame
            DataFrame to clean
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        self.df = df.copy()
        self.name = name
        
        # Find country code column
        col = [col for col in self.df.columns if col in ['Country Code', 'CountryCode']][0]
        
        # Remove invalid countries
        invalid_codes = self.fk_country(name, self.df, return_code=True, disp=False)
        mask = self.df[col].isin(invalid_codes)
        df_cleaned = self.df[~mask]
        
        # Save cleaned data
        self.attach_data(self.name, df_cleaned)
        self.saver.get_file(self.name, df_cleaned)
        self.saver.save_process_files(self.name, df_cleaned, path="processed", filetype='csv')
        
        return self