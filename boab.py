# -*- coding: utf-8 -*-
"""
load_data
check file extension if csv or xlsx or dat

fix column names (remove capitals, spaces, dots, symbols)

Fix data types to_numeric to_datetime

for each column check number of missing 
if more than 20% missing then remove
if missing records < 20% try impute
impute - KNN, random for labels or mean impute

Binning of contiuous vars

For categoricals make dummy variables

Always drop dummy [-1] for each group of variables

scale and normalise numeric variables

Balance data using smote

train test split

lassoCV, RidgeCV, Neural Net

Perf report

Save models


"""
import os
import pandas as pd 
import re
import numpy as np


class Boab(object):
    """
    Class Desc
    """
    def __init__(self, *args, **kwargs):
        self.df = pd.DataFrame([])
        self.time_cols = []
        self.date_cols = []
        self.models = []
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        return super().__init__(*args, **kwargs)
    
    def __str__(self):
        return str(self.df.head())
        
    def __repr__(self):
        return str(self.df.head())

    def head(self):
        print(self.df.head())
    
    @property
    def columns(self):
        return self.df.columns
        
    def fix_col_names(self):
        """
        fix column names (remove capitals, spaces, dots, symbols)
        """
        col_names = [] 
        for n in self.df.columns:
            coln = n.lower()
            coln = re.sub("([-_.\(\) ])", '_', coln)
            coln = re.sub("([+: ])", '', coln)
            col_names.append(coln)
        self.df.columns = col_names

    def load_data(self, filename, sep=','):
            """
            load_data
            check file extension if csv or xlsx or dat
            """
            ext = os.path.splitext(filename)[-1].replace('.','')
            if ext in ['csv', 'tsv', 'dat', 'data', 'txt', '']:
                # Check csv file to find separator
                import csv
                with open(filename, newline='') as csvfile:
                    dialect = csv.Sniffer().sniff(csvfile.read(1024))
                    separator = dialect.delimiter
                    print("\n[INFO] File separator: '{}'".format(separator))

                self.df = pd.read_csv(filename, sep=separator)
                # return self.df
            elif ext == 'xlsx' or ext == 'xls':
                self.df = pd.read_excel(filename)
                # return self.df
            self.fix_col_names()
        
    def __which_time_col(self, cols):
        # Look for date columns
        time_cols = []
        for i, c in enumerate(cols):
            match_obj = re.match("time", c, flags=re.IGNORECASE)
            if match_obj:
                time_cols.append(match_obj.group())
        return time_cols

    def __which_date_col(self, cols):
        # Look for date columns
        date_cols = []
        for i, c in enumerate(cols):
            match_obj = re.match("datetime|date|dte", c, flags=re.IGNORECASE)
            if match_obj:
                date_cols.append(match_obj.group())
        return date_cols
        

    def fix_types(self, french_decs=None):
        """
        Fix data types to_numeric, to_datetime
        Look at top 100 rows of each col and decide on a
        data type, if number rows < 100 then use all rows
        """        
 
        def fix_time_cols(self):
            # Fix TIme Columns
            try:
                # fix time cols
                for c in self.time_cols:
                    self.df[c] = pd.to_datetime(self.df[c], format='%H:%M:%S')
            except:
                pass

            # Fix TIme Columns
            try:
                # fix time cols
                for c in self.time_cols:
                    self.df[c] = pd.to_datetime(self.df[c], format='%H.%M.%S')
            except:
                pass
            
        def fix_date_cols(self):
            # fix date cols
            # for cols in date_cols do pd.to_datetime()
            for c in self.date_cols:
                try:
                    self.df[c] = pd.to_datetime(self.df[c])
                except:
                    self.df[c] = pd.to_datetime(self.df[c], format='%d/%m/%y %H.%M.%S')
                    
        ############################
        # FIX FRENCH DECIMALS
        ############################
        if french_decs:
            self.french_dec_english(french_decs)
        
        ############################
        # FIX DATE AND TIME COLUMNS
        ############################
        
        # Look for date columns
        self.date_cols = self.__which_date_col(self.df.columns)
        
        # fix date cols
        # for cols in date_cols do pd.to_datetime()
        # for c in self.date_cols:
        #     self.df[c] = pd.to_datetime(self.df[c])
        
        # fix date cols
        fix_date_cols(self)
        
        # Look for time columns
        self.time_cols = self.__which_time_col(self.df.columns)
        
        # Remove Date Cols from Time Cols
        self.time_cols = list(set(self.time_cols) - set(self.date_cols))
        
        # Fix time columns
        fix_time_cols(self)
        
        # Infer Objects
        self.df.infer_objects()
        
        
        # Do any columns contan 'nan'?
        # if so then fix
        for c in self.df.columns:
            # For string columns
            if self.df[c].dtype == object:
                print('Column:', c)
                mask = self.df[c] == 'nan'
                print('Number nans:', mask.shape[0])

                if mask.shape[0] > 0:
                    # Replace the nans
                    self.df.loc[mask, c] = np.NaN
                    try:
                        # Convert to numeric
                        self.df[c] = pd.to_numeric(self.df[c])
                    except:
                        pass
        
        return self
    
    def fix_missing(self):
        """
        For each col count the number of missing 
        divided by the number of rows. 
        Check if over 20% missing
        """
        n_rows = self.df.shape[0]
        perc_missing = []
        for c in self.df.columns:
            n_missing = self.df[c].isna().sum()
            perc_missing.append(n_missing/n_rows)
            
        drop_cols = []
        for i, p in enumerate(perc_missing):
            if p > 0.2:
                drop_cols.append(self.df.columns[i])
        
        # Drop the columns
        self.df.drop(drop_cols, axis=1, inplace=True)
        
        # If there are <20% missing then impute the column
        for c in self.df.columns:
            # Numeric Columns
            if(self.df[c].dtype == np.float64 or self.df[c].dtype == np.int64):
                print('Column:', c, 'Numeric')
                # for each numeric col do mean imputation
                self.df.loc[self.df[c].isnull(), c] = self.df[c].mean()
                                 
        return self
               
    def french_dec_english(self, colnames):
        """
        Convert 2,6 to 2.6
        """
        for c in colnames:
            self.df[c] = self.df[c].apply(lambda x: str(x).replace(',','.'))
        
        return self
    
    def is_time_componet(self, colname):
        # There is no time component
        if self.df[colname].dt.hour.sum() == 0 and self.df[colname].dt.minute.sum() == 0 and self.df[colname].dt.second.sum() == 0 and self.df[colname].dt.microsecond.sum() == 0:
            return False
        # There IS a time component
        else:
            return True
    
    def make_discrete_datetime_cols(self, verbose=False):
        # Look for date columns
        self.date_cols = self.__which_date_col(self.df.columns)
        # Look for time columns
        self.time_cols = self.__which_time_col(self.df.columns)
        
        if verbose:
            print('Date columns:', self.date_cols, 'Time columns', self.time_cols )
            
        for d in self.date_cols:
            # append col name as prefix
            self.df['_'.join([d, 'day'])] = self.df[d].dt.day
            self.df['_'.join([d, 'month'])] = self.df[d].dt.month
            self.df['_'.join([d, 'year'])] = self.df[d].dt.year
            
            # If there is a time component then extract hour
            if self.is_time_componet(d):
                self.df['_'.join([d, 'hour'])] = self.df[d].dt.hour
                
    def build_regression(self, feature_list, target, model_list=['ridge']):
        """
        Takes features and target
        Does train test split
        Reports on preformance 
        Returns: model
        """
        
        # Fix missing values
        print('\n[INFO] Imputing missing values')
        self.fix_missing()
        print('\nMissing Values:')
        print(self.df.isna().sum())
        
        seed = 4784
        
        from sklearn.model_selection import train_test_split
        X = self.df[feature_list]
        y = self.df[target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=seed)
        
        for m in model_list:
            if m == 'ridge':
                from sklearn.linear_model import RidgeCV
                # Choosing a CV number
                if self.df.shape[0] > 100:
                    cv = 3
                elif self.df.shape[0] > 500:
                    cv = 5
                else:
                    cv = 1
                
                model = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=cv)
                model.fit(self.X_train, self.y_train)
                print('\nRidge Regression R-squared:', model.score(self.X_test, self.y_test))
                # Add the model to the output list
                self.models.append(model)
                
    def build_ts_regression(self, feature_list, target, dt_index, model_list=['ridge']):
        """
        Takes features and target
        Does train test split
        Reports on preformance 
        Returns: model
        """
        
        # Fix missing values
        print('\n[INFO] Imputing missing values')
        self.fix_missing()
        print('\nMissing Values:')
        print(bo.df.isna().sum())
        
        test_size = 0.3
        
        self.df.sort_values(dt_index, ascending=True, inplace=True)
        nrows = self.df.shape[0]
        train_idx = int(nrows*(1-test_size))
        test_idx = nrows - train_idx
                
        X = self.df[feature_list]
        y = self.df[target]
        
        self.X_train = X.iloc[0:train_idx]
        self.X_test =  X.iloc[0:test_idx]    
        self.y_train = y.iloc[0:train_idx]
        self.y_test =  y.iloc[0:test_idx]    
        
        print('Xtrain size:', self.X_train.shape[0], 'Xtest size:', self.X_test.shape[0])
        
        for m in model_list:
            if m == 'ridge':
                from sklearn.linear_model import RidgeCV
                # Choosing a CV number
                if self.df.shape[0] > 100:
                    cv = 3
                elif self.df.shape[0] > 500:
                    cv = 5
                else:
                    cv = 1
                
                model = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=cv)
                model.fit(self.X_train, self.y_train)
                print('\nRidge Regression R-squared:', model.score(self.X_test, self.y_test))    
                # Add the model to the output list
                self.models.append(model)
        
                                 
############################################################
### End Class
############################################################
                                 
############################################################
### Example API
############################################################
# bo = Boab()
# bo.load_data('boab-data-science/boab/AirQualityUCI.csv', sep=';')
# bo.fix_types(french_decs=['co_gt_', 'c6h6_gt_', 't', 'rh', 'ah'])
# bo.fix_missing()
# # bo.french_dec_english(['co_gt_', 'c6h6_gt_', 't', 'rh', 'ah'])
# bo.make_discrete_datetime_cols()
# bo.df.head()

# # Build normal regression
# feature_list = ['pt08_s1_co_', 'nmhc_gt_', 'c6h6_gt_',
#        'pt08_s2_nmhc_', 'nox_gt_', 'pt08_s3_nox_', 'no2_gt_', 'pt08_s4_no2_',
#        'pt08_s5_o3_', 't', 'rh', 'ah', 'date_day', 'date_month', 'date_year']
# target = 'co_gt_'
# bo.build_regression(feature_list=feature_list, target=target)

# # Build Timeseries regression
# feature_list = ['pt08_s1_co_', 'nmhc_gt_', 'c6h6_gt_',
#        'pt08_s2_nmhc_', 'nox_gt_', 'pt08_s3_nox_', 'no2_gt_', 'pt08_s4_no2_',
#        'pt08_s5_o3_', 't', 'rh', 'ah', 'date_day', 'date_month', 'date_year']
# target = 'co_gt_'
# bo.build_ts_regression(feature_list=feature_list, target=target, dt_index='date')

############################################################
## CAR DATA
############################################################

bo = Boab()
bo.load_data('car_data.csv')
bo.fix_types()
bo.fix_missing()
bo.make_discrete_datetime_cols()
print(bo.df.head())

# Build normal regression
feature_list = ['cyclinders', 'displacement', 'hp', 'weight', 'acceleration', 'year', 'origin']
target = 'mpg'
bo.build_regression(feature_list=feature_list, target=target)