from pubchempy import get_compounds
from rdkit import Chem
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_descriptors(categorical_dataframe,
                         dataframe_to_be_concat_with):

    # selects unique formulas
    compounds = list(categorical_dataframe['material'].unique())

    # get smiles from formulas
    smiles = []
    for compound in compounds:
        for compound_obj in get_compounds(compound, 'name'):
            smiles.append(compound_obj.canonical_smiles)

    # generates descriptors as a list (each mol) of lists (each desc); converts to dataframe
    generated_descriptors = []
    descriptors_names = ['test']
    for one_smile in smiles:
        mol = Chem.MolFromSmiles(one_smile)
        descriptors_for_one_mol = [1]
        generated_descriptors.append(descriptors_for_one_mol)
    descriptors_df = pd.DataFrame(generated_descriptors, columns=descriptors_names)

    # normalizes the generated descriptors; converts to dataframe; concatenates with compounds
    names = descriptors_df.columns
    sc = MinMaxScaler()
    descriptors_df_norm = pd.DataFrame(sc.fit_transform(descriptors_df), columns=names)
    concatenated = pd.concat([descriptors_df_norm, pd.DataFrame(compounds,
                                                                columns=['material'])], axis=1)
    concatenated = concatenated.iloc[:20, :]

    return pd.merge(dataframe_to_be_concat_with, concatenated)
