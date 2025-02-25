import pandas as pd
from typing import List
from ..discrete_problem import ChemistryDatasetLoader

class DenmarkLoader(ChemistryDatasetLoader):
    """
    Dataset loader for the Denmark dataset.
    """
    def __init__(self, variable_names: List[str], target: str, enhance: bool = True, prefix: str = ""):
        super().__init__(dataset_name="Denmark")
        self.model_path = prefix + 'src/currybo/test_functions/chemistry_datasets/Denmark.pkl'
        if enhance:
            self.csv_path = prefix + 'src/currybo/test_functions/chemistry_datasets/Denmark_enhanced.csv'
        else:
            self.csv_path = prefix + 'src/currybo/test_functions/chemistry_datasets/Denmark_cleaned.csv'

        self.load_data()
        self.preprocess_data(variable_names, target)
        self.enhanced = enhance
        self.target = target

    def load_data(self):
        self.dataset = pd.read_csv(self.csv_path, index_col = False)

    def preprocess_data(self, variable_names: List[str], target: str):
        if not all(variable == "Catalyst" for variable in variable_names):
            raise ValueError("Invalid variable for dataset")

        unique_smiles = self.dataset["Catalyst"].unique().tolist()
        fp_list = [self.smiles_to_fingerprint(smiles) for smiles in unique_smiles]
        self.x_options["Catalyst"] = fp_list

        if target != "Delta_Delta_G":
            raise ValueError("Invalid target for dataset")

        self.max_value = self.dataset['Delta_Delta_G'].max() * 1.5
        self.min_value = self.dataset['Delta_Delta_G'].min() * 1.5

        w_columns = [col for col in self.dataset.columns if col not in (variable_names + [target]) and col != self.dataset.index.name]

        for column in w_columns:
            unique_smiles = self.dataset[column].unique().tolist()
            fp_list = [self.smiles_to_fingerprint(smiles) for smiles in unique_smiles]
            self.w_options[column] = fp_list
