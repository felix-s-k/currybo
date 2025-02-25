<table style="border-collapse: collapse; border: none; text-align: center; background: none; border-spacing: 0; padding: 0;">
  <tr style="background: none; border: none;">
    <!-- Logo row spanning three columns -->
    <td colspan="3" style="border: none; background: none; padding: 0;">
      <img src="./CurryBO-logo.svg" alt="CurryBO-logo" width="300px">
    </td>
  </tr>
  <tr style="background: none; border: none;">
    <!-- License badge -->
    <td style="border: none; background: none; padding: 0;">
      <a href="https://gitlab.com/aspuru-guzik-group/self-driving-lab/instruments/chemspyd/-/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
      </a>
    </td>
    <!-- Python version badge -->
    <td style="border: none; background: none; padding: 0;">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python Version">
      </a>
  </tr>
</table>

# CurryBO: Bayesian optimization over curried function spaces

`CurryBO` is a pure python package that allows to conduct Bayesian Optimization in the search for general (i.e. transferable) parameters that *work well* across multiple related tasks.

## Installation

To install the package, simply:
```
git clone https://github.com/felix-s-k/currybo.git
cd currybo
pip install -e .
```

## Usage

Example use cases to identify general reaction conditions in chemistry are shown in the `benchmarks` folder for two different aggregation functions, the mean aggregation function and threshold aggregation function. This folder also contains all the files to reproduce the plots in the publication - to reproduce the benchmarks, run all the python files in the `benchmark_mean` and `benchmark_frac` folders.
In order to run the benchmarks, you first need to create the oracle (a `.pkl` file), for that run the Jupyter Notebook in the folder `data-analysis/dataset-name/dataset-name_emulator.ipynb`. This will produce a `.pkl` file, move it to `src/currybo/test_functions/chemistry_datasets/`.


