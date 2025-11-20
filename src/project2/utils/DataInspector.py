import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats 
import pandas as pd  
import textwrap
from IPython.display import display, Markdown
import missingno as msno
import math 
import numpy as np 

from project2.data.Saver import Saver
from project2.data.Filesloader import Reader


class BasicInfo:
    def __init__(self, reader:Reader,saver=None):

        self.reader = reader
        self.df = None
        self.name = None
        self.numerical_feat = []
        self.categorical_feat = []
        self.saver = saver
        self.prodata = {}

    @property
    def data_dict(self):
        """Always return the current data_dict from the Reader."""
        return self.reader.data_dict


        

    def attach_data(self, df=None):
        """
            Attaching the data files 
        """
        if df is not None:
            self.df = df

        self.numerical_feat, self.categorical_feat = self.featuretypes()
        
        if self.saver is None:
            self.saver = Saver()
       
        return self.name, self.df 
    
       
    

    def show_head(self):
        for idx, (name, df) in enumerate(self.data_dict.items()):
            self.name = name
            self.df = df
            self.attach_data()
            
            display(Markdown(f"### ----------- {idx} - {self.name} --------------"))
            display(self.df.head(5))
            
            

    def report(self):
        report_dict = {
            "shape": [self.df.shape],
            "columns": [len(self.df.columns)],
            "rows": [len(self.df)],
            "missing (%)": [self.df.isna().mean().mean() * 100],
            "numeric_features": [len(self.numerical_feat)],
            "categorical_features": [len(self.categorical_feat)],
            "Duplicants": [self.df.duplicated().sum()]
        }
        return pd.DataFrame(report_dict)
    
    

    def featuretypes(self, threshold_num=10):

        """Returns categorical or numerical data"""

        numerical_feat, categorical_feat = [], []
        for feat in self.df.columns:
            dtype = self.df[feat].dtype
            if pd.api.types.is_bool_dtype(dtype):
                categorical_feat.append(feat)
            elif pd.api.types.is_numeric_dtype(dtype):
                if self.df[feat].nunique() <= threshold_num:
                    categorical_feat.append(feat)
                else:
                    numerical_feat.append(feat)
            else:
                categorical_feat.append(feat)
        return numerical_feat, categorical_feat
    



    def info(self):
        """
        show basic info for each file
        """

        for idx, (name, df) in enumerate(self.data_dict.items()):
            self.name = name
            self.df = df
            self.attach_data()
            
            display(Markdown(f"### --- Info-------- {idx} - {self.name} --------------"))
            display(Markdown(df.info()))
            display(self.report())

    
            
    def describe(self):
        for idx, (name, df) in enumerate(self.data_dict.items()):
            self.name = name
            self.df = df
            self.attach_data()
            
            dff = df[self.numerical_feat].dropna()
            if dff.empty:
                display(Markdown(f"### No numerical features here - {self.name}"))
            else: 
                describe = dff.describe().T
                display(Markdown(f"### --- Description-------- {idx} - {self.name} --------------"))
                display(describe)
    



    def missingness(self, show_all=True, return_data=False, disp=False):

        """
        Display and optionally return missingness per dataset.
        Returns a dict of missingness Series if return_data=True.
        """

        all_missing = {}

        for idx, (name, df) in enumerate(self.data_dict.items()):
            self.name = name
            self.df = df
            self.attach_data()

            display(Markdown(f"### --- Missingness -------- {idx} - {self.name} --------------"))
            df_miss = (df.isna().mean() * 100).to_frame("%Missing").round(2)
            all_missing[name] = df_miss["%Missing"]

            if disp:
                # display section
                if show_all:
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        display(df_miss)
                else:
                    display(df_miss.head(15)) 
            else:None

        if return_data:
            return all_missing
        return None
    



    def dropabove(self, skip_cols=None, missing_threshold=70):
        """
        Drops columns with missingness above a given threshold across all datasets.
        """
        all_missing = self.missingness(return_data=True)
        
        for name, miss in all_missing.items():
            df = self.data_dict[name].copy()
            cols_to_drop = miss[miss > missing_threshold].index
            
            # Keep some columns if needed
            if skip_cols is not None and len(skip_cols) > 0:
                cols_to_drop = [c for c in cols_to_drop if c not in skip_cols]
            
            # Drop columns and update dictionary
            if len(cols_to_drop) > 0:
                df = df.drop(columns=cols_to_drop)
                display(Markdown(f"Dropped {len(cols_to_drop)} columns from **{name}** "
                                f"(>{missing_threshold}% missing): {list(cols_to_drop)}"))
            else:
                display(Markdown(f"No columns dropped in **{name}** — all below {missing_threshold}%."))
            
            # Update data dictionary
            self.data_dict[name] = df
            
            # Save to memory and disk
            self.saver.get_file(name, df)
            self.saver.save_process_files(name, df, path='interim')

            #display(Markdown("###  Column cleaning complete"))





    def visualize_missingness(self, ncols=2):
        n = len(self.data_dict)
        nrows = (n + ncols - 1) // ncols

        # Increase figure size dynamically for better spacing
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 6 * nrows))
        ax = ax.flatten()

        for i, (name, df) in enumerate(self.data_dict.items()):
            axis = ax[i]
            self.name = name
            self.df = df
            self.attach_data()

            # Skip if no missing data
            if not df.isna().any().any():
                display(Markdown(f" No missing data in **{self.name}**"))
                axis.set_visible(False)
                continue

            display(Markdown(f"### Missingness — {i+1}: {self.name}"))

            msno.matrix(df, ax=axis, sparkline=False, fontsize=10, color=(0.1, 0.3, 0.8))
            
            axis.set_title(f"{name}", fontsize=14, pad=15)
            axis.tick_params(axis='x', labelrotation=45, labelsize=9)
            axis.tick_params(axis='y', labelsize=8)

        # Hide unused axes (if total < grid size)
        for j in range(i + 1, len(ax)):
            ax[j].set_visible(False)

        plt.tight_layout(pad=3)
        plt.show()




    
    def modality(self, show_all=True, return_value=False, disp=True):

        """Handling categorical features """

        for name, df in self.data_dict.items(): 
            
            self.name = name 
            self.df = df 
            self.attach_data()
            
            if not self.categorical_feat:
                return "No categorical features"
            
            
            
            nunique = self.df[self.categorical_feat].nunique()
            
            df_nunique = nunique.to_frame('Nunique Features')

            if disp:
                if show_all:
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        display(f'---------{name}---------------')
                        display(df_nunique)
                else: None
            else: None
            
        if return_value:
            return nunique
        else: return None 
        



    def modalclassification(self):
        for name, df in self.data_dict.items():
            print(name, '', df)
            self.name = name 
            self.df = df 
            self.attach_data()
            data_ranges = [(2, 10, []), (11, 20, []), (21, 30,[]), (31,40, [])] 

            categ_feat = self.categorical_feat
            for col in categ_feat:
                for min_nunique, max_nunique, nunique_list in data_ranges:
                    nunique = self.df[col].nunique()
                    if min_nunique <= nunique <= max_nunique:
                        tab = self.modalityquantytab(col) 
                        nunique_list.append(tab)

            self.prodata[name] = data_ranges    

        return None  



        
    def modalityquantytab(self,col):
        value_counts =  self.df[col].value_counts()
        modality = value_counts.index
        tab = pd.DataFrame(modality)
        tab['n'] = value_counts.values
        tab['f'] = self.df[col].\
            value_counts(normalize=True).values.round(2)
        tab['F'] = tab['f'].cumsum()
        return tab 
        


    def dataTable(self, dataset_name=None):

        for tab_name, range_list in self.prodata.items():
            if dataset_name and tab_name != dataset_name:
                continue

            print(f"\n### Dataset: {tab_name}")

            for (min_val, max_val, df_list) in range_list:
                if not df_list:  # skip empty ranges
                    continue

                n_tables = len(df_list)
                ncols = 2
                nrows = math.ceil(n_tables / ncols)

                # Bigger, adaptive figure size
                fig, axes = plt.subplots(
                    nrows=nrows, ncols=ncols,
                    figsize=(8 * ncols, 4 * nrows)
                )
                axes = axes.flatten()

                for i, (tab, ax) in enumerate(zip(df_list, axes)):
                    x = tab.iloc[:, 0].astype(str)
                    y = tab['f']

                    # Horizontal bars to fit long text
                    ax.barh(x, y, color='skyblue', edgecolor='black')
                    ax.set_title(f"{tab.iloc[:, 0].name} ({len(x)} categories)", fontsize=11, pad=8)
                    ax.set_xlabel("Frequency (f)")
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', labelsize=8)
                    ax.invert_yaxis()  # most frequent at top

                # Hide unused axes
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)

                # Global title & layout
                fig.suptitle(
                    f"{tab_name} — {min_val}–{max_val} unique values",
                    fontsize=16, weight='bold', y=1.02
                )
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                plt.show()

    

class DummyReader:
    """
    A lightweight wrapper to mimic the Reader class structure expected by BasicInfo.
    It provides a .data_dict attribute so BasicInfo can access datasets normally.
    """
    def __init__(self, data_dict):
        self.data_dict = data_dict 


    




    



class VisualInspector:
    """This class is for visualisation of basic details of the data"""
    def __init__(self, inspect:BasicInfo):
        self.df = inspect.df
        self.data = {}
        self.numerical_feat = inspect.numerical_feat
        self.categorical_feat = inspect.categorical_feat
        self.name = inspect.name
        self.tab = None

        #self.modalclassification()



    def visualisenumericsfeat(self, figsize=(15, 5)):
        numeric_cols = self.numerical_feat 
        print(f"The numerical featues: {numeric_cols}")
        if not numeric_cols:
            print("No numeric columns found.")
            return

        for col in numeric_cols:
            fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=120)
            fig.suptitle(f"{col}", fontsize=14, fontweight="bold")
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[0], color="skyblue")
            axes[0].set_title("Histogram", fontsize=12)
            axes[0].set_xlabel(col)
            axes[0].set_ylabel("Frequency")

            sns.boxplot(data=self.df, x=col, ax=axes[1], color="lightgreen",
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=5))
            axes[1].set_title("Boxplot", fontsize=12)
            axes[1].set_xlabel(col)

            stats.probplot(self.df[col].dropna(), dist="norm", plot=axes[2])
            axes[2].set_title("Q-Q Plot", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()




    def categ_plot_collection_bars(self, collection, ncols=2, figsize=(16, 10)):
        n = len(collection)
        nrows = (n + ncols - 1)//ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for ax, (key,feature) in zip(axes,collection.items()):
            feature = feature.sort_values("f", ascending=False)
            if key in ['Source', 'Topic']:
                feature[key] = feature[key].apply(lambda x: textwrap.fill(x, 40))
                if 'Source':
                    feature = feature.head(20)
            sns.barplot(data=feature, x='f', y=key, ax=ax)
            ax.set_xlabel('Frequency')
            ax.set_ylabel(f'{key.title()}')
            ax.set_title(f'{key.title()}')
            plt.subplots_adjust(hspace=0.8, wspace=0.6)
            plt.tight_layout(rect=[0, 0, 1, 1])
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
        for ax in axes[len(collection):]:
            ax.set_visible(False)
        plt.show()




    