import pandas as pd
from typing import List
from ..discrete_problem import ChemistryDatasetLoader

class BorylationLoader(ChemistryDatasetLoader):
    """
    Dataset loader for the borylation dataset.
    """
    def __init__(self, variable_names: List[str], target: str, enhance: bool = True, prefix: str = ""):
        super().__init__(dataset_name="Borylation")
        self.model_path = prefix + 'src/currybo/test_functions/chemistry_datasets/borylation.pkl'
        if enhance:
            self.csv_path = prefix + 'src/currybo/test_functions/chemistry_datasets/borylation_enhanced.csv'
        else:
            self.csv_path = prefix + 'src/currybo/test_functions/chemistry_datasets/borylation_cleaned.csv'

        self.load_data()
        self.preprocess_data(variable_names, target)
        self.enhanced = enhance
        self.target = target

    def load_data(self):
        self.dataset = pd.read_csv(self.csv_path, index_col = False)

    def preprocess_data(self, variable_names: List[str], target: str):
        valid_variables = {"ligand", "solvent"}
        if any(variable not in valid_variables for variable in variable_names):
            raise ValueError("Invalid variable for dataset")
        
        predefined_variable_names = ["ligand", "solvent"]

        if variable_names != predefined_variable_names:
            print("Reordered variables are used to be compatible with the ML model!")
            variable_names = predefined_variable_names
        
        for variable in variable_names:
            unique_smiles = self.dataset[variable].unique().tolist()
            fp_list = [self.smiles_to_fingerprint(smiles) for smiles in unique_smiles]
            self.x_options[variable] = fp_list

        if target != "yield":
            raise ValueError("Invalid target for dataset")

        self.max_value = 1
        self.min_value = 0

        w_columns = [col for col in self.dataset.columns if col not in (variable_names + [target]) and col != self.dataset.index.name]

        for column in w_columns:
            unique_smiles = self.dataset[column].unique().tolist()
            fp_list = [self.smiles_to_fingerprint(smiles) for smiles in unique_smiles]
            self.w_options[column] = fp_list