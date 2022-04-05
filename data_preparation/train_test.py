import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import category_encoders as ce
import numpy as np
from sklearn.model_selection import train_test_split

from data_preparation.descriptors_generation import generate_descriptors


def make_train_test(bins_interval=[0, 100],
                    bins_step=1,
                    test_size=0.2,
                    random_state=2022):

    # loads numeric and categorical part of the final dataset
    current_path = os.getcwd()
    upper_directory = '\\'.join(current_path.split('\\')[0:-1])
    db_numeric = pd.read_csv(upper_directory + '\\databases\\Final\\numerical.csv')
    db_categorical = pd.read_csv(upper_directory + '\\databases\\Final\\categorical.csv')

    # splits the data on predictors and value to be predicted
    y_final = db_numeric.loc[:, db_numeric.columns == 'viability (%)']
    x_numeric = db_numeric.loc[:, db_numeric.columns != 'viability (%)']

    # scales the numeric data from 0 to 1 for each column; converts to dataframe
    sc = MinMaxScaler()
    x_numeric = pd.DataFrame(sc.fit_transform(x_numeric),
                             columns=['del',
                                      'time (hr)',
                                      'concentration (ug/ml)',
                                      'Hydrodynamic diameter (nm)',
                                      'Zeta potential (mV)'])
    del x_numeric['del']

    # encodes categorical data; converts to dataframe; adds column with material (for next merge only)
    materials = db_categorical['material']
    db_categorical.drop(['material'], axis=1)
    encoder = ce.OrdinalEncoder(return_df=True)
    x_categorical = pd.DataFrame(encoder.fit_transform(db_categorical))
    del x_categorical['Unnamed: 0']
    x_cat_plus_mat = x_categorical.assign(material=materials)

    # forms dataframe with predictors (categorical + numeric + material_column)
    x = pd.concat([x_cat_plus_mat, x_numeric], axis=1)

    # final predictors
    x = generate_descriptors(categorical_dataframe=db_categorical,
                             dataframe_to_be_concat_with=x)

    # drops some empty values (if present)
    x_y = pd.concat([x, y_final], axis=1)
    x_y = x_y.dropna()
    del x_y['material']

    # final split; makes bins to split data based on continuous value (viability)
    y_final = x_y.loc[:, 'viability (%)']
    x_final = x_y.loc[:, x_y.columns != 'viability (%)']

    bins = np.linspace(bins_interval[0],
                       bins_interval[1],
                       bins_step)
    y_binned = np.digitize(y_final, bins)

    x_train, x_test, y_train, y_test = train_test_split(x_final,
                                                        y_final,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y_binned)

    return x_train, x_test, y_train, y_test

