from argparse import ArgumentParser
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score


def setup_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Modred app")
    parser.add_argument("--filename", "-f", type=str, help="Name of the file to process, this file must be .csv and it contains two columns molecules as SMILES format and activity value.")
    parser.add_argument("--output", "-o", type=str, default='output.csv', help="Name of the output file")
    parser.add_argument("--3D", type=bool, default=False, help="Using 3D molecular descriptor. (default = False)")
    parser.add_argument("--maxvar", type=int, default=4, help="Number of maximum variable in QSAR model. (Default = 4)")
    parser.add_argument("--maxmodel", type=int, default=1000, help="Number of maximum model to report. (Default = 1000)")
    parser.add_argument("--interval", type=int, default=3, help="Interval of data for train-test splitting data (Default = 3)")
    parser.add_argument("--r2_filter", type=float, default=0.3, help="R^2 score for filtering out of uncorrelated descriptors (Default = 0.3)")
    parser.add_argument("--pair_filter", type=float, default=0.7, help="R^2 score for filtering out of correlated descriptors (Default = 0.7)")
    parser.add_argument("--wo_outlier", type=bool, default=False, help="Building QSAR without outlier data. (Default = False)")
    parser.add_help = True
    return parser

def processing_3d(molecule: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    molecule: Chem.rdchem.Mol = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule)
    return molecule

def calculate_descriptor(molecules: list[Chem.rdchem.Mol], use3D: bool = False) -> pd.DataFrame:
    if use3D:
        molecules: list[Chem.rdchem.Mol] = [processing_3d(mol) for mol in molecules]
    calc: Calculator  = Calculator(descriptors, ignore_3D = not use3D)
    des: pd.DataFrame = calc.pandas(molecules).select_dtypes(exclude=['object'])
    des.index         = [f'mol_{i+1}' for i in range(len(molecules))]
    return des 

def clean_onevar(x: pd.DataFrame, y: pd.DataFrame, r2_criteria: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores: np.ndarray     = np.array([LinearRegression().fit(x[des].values.reshape(-1, 1), y.values.reshape(-1, 1)).score(x[des].values.reshape(-1, 1), y.values.reshape(-1, 1)) for des in x.columns])
    new_columns: list[str] = x.columns[np.argsort(scores)[::-1]]
    new_columns            = new_columns[np.sort(scores)[::-1] > r2_criteria]
    new_x: pd.DataFrame    = x[new_columns]
    new_x                  = new_x.iloc[np.argsort(y)]
    new_y: pd.DataFrame    = y.sort_values(axis=0)
    return new_x, new_y

def train_test_split(x: pd.DataFrame, y: pd.DataFrame, interval: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_index       = np.arange(stop=len(x), step=interval)
    train_index      = np.delete(np.arange(len(x)), test_index)
    x_train, y_train = x.iloc[train_index], y.iloc[train_index]
    x_test, y_test   = x.iloc[test_index], y.iloc[test_index]
    return x_train, y_train, x_test, y_test

def check_self_corelation(var_name: list[str], self_corelation: dict[tuple[str, str], float], criteria: float) -> bool:
    return np.array([self_corelation[pair] < criteria for pair in combinations(var_name, 2)]).all()

def create_header(max_var: int) -> pd.DataFrame:
    coef_name_df: pd.DataFrame  = pd.DataFrame({f"Feature_{i+1}":[] for i in range(max_var)})
    coef_value_df: pd.DataFrame = pd.DataFrame({f"Coefficient_{i+1}":[] for i in range(max_var)})
    intercept_df: pd.DataFrame  = pd.DataFrame({"Intercept":[]})
    scores_df: pd.DataFrame     = pd.DataFrame({"R2_training":[], "R2_Test":[], "Q2_Score":[]})
    outlier_df: pd.DataFrame    = pd.DataFrame({"Outlier":[]})
    result_df: pd.DataFrame     = pd.concat([coef_name_df, coef_value_df, intercept_df, scores_df, outlier_df], axis=1)
    return result_df


def build_model(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, des_names: list[str], num_vars: int = 1) -> pd.DataFrame:
    if num_vars == 1:
        lr_model: LinearRegression  = LinearRegression()
        q2_scores: float            = 1 - np.abs(cross_val_score(lr_model, x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1), cv=LeaveOneOut(), scoring='neg_mean_squared_error')).sum()/((y_train.values - y_train.values.mean())**2).sum()
        lr_model.fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
        r2_train: float             = lr_model.score(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
        r2_test: float              = r2_score(y_test.values.reshape(-1, 1), lr_model.predict(x_test.values.reshape(-1, 1)))
    else:
        lr_model: LinearRegression  = LinearRegression()
        q2_scores: float            = 1 - np.abs(cross_val_score(lr_model, x_train.values, y_train.values.reshape(-1, 1), cv=LeaveOneOut(), scoring='neg_mean_squared_error')).sum()/((y_train.values - y_train.values.mean())**2).sum()
        lr_model.fit(x_train.values, y_train.values.reshape(-1, 1))
        r2_train: float             = lr_model.score(x_train.values, y_train.values.reshape(-1, 1))
        r2_test: float              = r2_score(y_test.values, lr_model.predict(x_test.values))
    coef_name_df: pd.DataFrame      = pd.DataFrame({f"Feature_{i+1}":[des] for i, des in enumerate(des_names)})
    coef_value_df: pd.DataFrame     = pd.DataFrame({f"Coefficient_{i+1}":[val] for i, val in enumerate(lr_model.coef_[0])})
    intercept_df: pd.DataFrame      = pd.DataFrame({"Intercept":lr_model.intercept_})
    scores_df: pd.DataFrame         = pd.DataFrame({"R2_training":[r2_train], "R2_Test":[r2_test], "Q2_Score":[q2_scores]})
    outlier_df: pd.DataFrame        = pd.DataFrame({"Outlier":[]})
    result_df: pd.DataFrame         = pd.concat([coef_name_df, coef_value_df, intercept_df, scores_df, outlier_df], axis=1)
    return result_df

def build_model_wo_outlier(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, des_names: list[str], num_vars: int = 1) -> pd.DataFrame:
    original_index: list[str] = [mol_index for mol_index in x_train.index]
    outliers: list[str]       = []
    new_x: pd.DataFrame       = deepcopy(x_train)
    new_y: pd.DataFrame       = deepcopy(y_train)
    models: pd.DataFrame      = create_header(num_vars)
    while True:
        if num_vars == 1:
            lr_model: LinearRegression     = LinearRegression()
            lr_model.fit(new_x.values.reshape(-1, 1), new_y.values.reshape(-1, 1))
            old_r2: float                  = lr_model.score(new_x.values.reshape(-1, 1), new_y.values.reshape(-1, 1))
            residual: float                = (new_y - lr_model.predict(new_x.values.reshape(-1, 1)).flatten())**2
            outlier_index: int             = np.argmax(residual)
            outliers.append(original_index[outlier_index])
            new_x: pd.DataFrame            = new_x.drop(index=original_index[outlier_index])
            new_y: pd.DataFrame            = new_y.drop(index=original_index[outlier_index])
            original_index.pop(outlier_index)
            new_lr_model: LinearRegression = LinearRegression()
            q2_score: float                = 1 - np.abs(cross_val_score(new_lr_model, new_x.values.reshape(-1, 1), new_y.values.reshape(-1, 1), cv=LeaveOneOut(), scoring='neg_mean_squared_error')).sum()/((new_y - new_y.mean())**2).sum()
            new_lr_model.fit(new_x.values.reshape(-1, 1), new_y.values.reshape(-1, 1))
            r2_new: float                  = new_lr_model.score(new_x.values.reshape(-1, 1), new_y.values.reshape(-1, 1))
            r2_test: float                 = r2_score(y_test.values.reshape(-1, 1), new_lr_model.predict(x_test.values.reshape(-1, 1)))
            coef_name_df: pd.DataFrame     = pd.DataFrame({f"Feature_{i+1}":[des] for i, des in enumerate(des_names)})
            coef_value_df: pd.DataFrame    = pd.DataFrame({f"Coefficient_{i+1}":[val] for i, val in enumerate(new_lr_model.coef_[0])})
            intercept_df: pd.DataFrame     = pd.DataFrame({"Intercept":new_lr_model.intercept_})
            scores_df: pd.DataFrame        = pd.DataFrame({"R2_training":[r2_new], "R2_Test":[r2_test], "Q2_Score":[q2_score]})
            outlier_df: pd.DataFrame       = pd.DataFrame({"Outlier":[outliers]})
            result_df: pd.DataFrame        = pd.concat([coef_name_df, coef_value_df, intercept_df, scores_df, outlier_df], axis=1)
            models: pd.DataFrame           = pd.concat([models, result_df])
            if r2_new < old_r2*1.1:
                break
        else:
            lr_model: LinearRegression     = LinearRegression()
            lr_model.fit(new_x.values, new_y.values.reshape(-1, 1))
            old_r2: float                  = lr_model.score(new_x.values, new_y.values.reshape(-1, 1))
            residual: float                = (new_y - lr_model.predict(new_x.values).flatten())**2
            outlier_index: int             = np.argmax(residual)
            outliers.append(original_index[outlier_index])
            new_x: pd.DataFrame            = new_x.drop(index=original_index[outlier_index])
            new_y: pd.DataFrame            = new_y.drop(index=original_index[outlier_index])
            original_index.pop(outlier_index)
            new_lr_model: LinearRegression = LinearRegression()
            q2_score: float                = 1 - np.abs(cross_val_score(new_lr_model, new_x.values, new_y.values.reshape(-1, 1), cv=LeaveOneOut(), scoring='neg_mean_squared_error')).sum()/((new_y - new_y.mean())**2).sum()
            new_lr_model.fit(new_x.values, new_y.values.reshape(-1, 1))
            r2_new: float                  = new_lr_model.score(new_x.values, new_y.values.reshape(-1, 1))
            r2_test: float                 = r2_score(y_test.values.reshape(-1, 1), new_lr_model.predict(x_test.values))
            coef_name_df: pd.DataFrame     = pd.DataFrame({f"Feature_{i+1}":[des] for i, des in enumerate(des_names)})
            coef_value_df: pd.DataFrame    = pd.DataFrame({f"Coefficient_{i+1}":[val] for i, val in enumerate(new_lr_model.coef_[0])})
            intercept_df: pd.DataFrame     = pd.DataFrame({"Intercept":new_lr_model.intercept_})
            scores_df: pd.DataFrame        = pd.DataFrame({"R2_training":[r2_new], "R2_Test":[r2_test], "Q2_Score":[q2_score]})
            outlier_df: pd.DataFrame       = pd.DataFrame({"Outlier": [outliers]})
            result_df: pd.DataFrame        = pd.concat([coef_name_df, coef_value_df, intercept_df, scores_df, outlier_df], axis=1)
            models: pd.DataFrame           = pd.concat([models, result_df])
            if r2_new < old_r2*1.1:
                break
    return models


def main() -> None:
    args             = setup_argparser().parse_args()
    all_vars: dict   = vars(args)
    df: pd.DataFrame = pd.read_csv(all_vars['filename'])
    df.index         = [f'mol_{i+1}' for i in range(len(df))]
    x: pd.DataFrame  = calculate_descriptor([Chem.MolFromSmiles(smiles) for smiles in df[df.columns[0]]], use3D=all_vars['3D'])
    y: pd.DataFrame  = df[df.columns[-1]]
    new_x, new_y     = clean_onevar(x, y, all_vars['r2score'])
    self_corelation  = {com:LinearRegression().fit(new_x[com[0]].values.reshape(-1, 1), new_x[com[1]].values.reshape(-1, 1)).score(new_x[com[0]].values.reshape(-1, 1), new_x[com[1]].values.reshape(-1, 1)) for com in combinations(new_x.columns, 2)}
    x_train, y_train, x_test, y_test = train_test_split(new_x, new_y, all_vars['interval'])
    header: pd.DataFrame = create_header(all_vars['maxvar'])
    for numvar in range(all_vars['maxvar']):
        if numvar+1 == 1:
            results: pd.DataFrame = pd.concat([header, pd.concat([build_model(x_train[des], y_train, x_test[des], y_test, [des], num_vars=numvar+1) for des in x_train.columns], ignore_index=True)], ignore_index=True)
            if not all_vars['wo_outlier']:
                results: pd.DataFrame = pd.concat([results, pd.concat([build_model_wo_outlier(x_train[des], y_train, x_test[des], y_test, [des], num_vars=numvar+1) for des in x_train.columns], ignore_index=True)], ignore_index=True)
        else:
            results: pd.DataFrame = pd.concat([results, pd.concat([build_model(x_train[list(des)], y_train, x_test[list(des)], y_test, des, num_vars=numvar+1) for des in combinations(x_train.columns, numvar+1) if check_self_corelation(des, self_corelation, all_vars['pairscore'])], ignore_index=True)], ignore_index=True)
            if not all_vars['wo_outlier']:
                results: pd.DataFrame = pd.concat([results, pd.concat([build_model_wo_outlier(x_train[list(des)], y_train, x_test[list(des)], y_test, des, num_vars=numvar+1) for des in combinations(x_train.columns, numvar+1) if check_self_corelation(des, self_corelation, all_vars['pairscore'])], ignore_index=True)], ignore_index=True)
    results: pd.DataFrame = results.iloc[range(all_vars['maxmodel'])]
    results.to_csv(all_vars['output'])

if __name__ == "__main__":
    main()