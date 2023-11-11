# ENCODiT

### Error flagging and Neutralization using Conformal Out-of-Distribution Detection in Time-Series Data

CS637A Course Project

November, 2023

### Group - 14

1. Kartik Anant Kulkarni (210493)
2. Rishi Agarwal (210849)
3. Emaad Ahmed (210369)
4. Dhruva Singh Sachan (210343)

## How to Run

- Add the dataset to `./dataset` folder as per the organisation specified in `./dataset/Dataset.md`.
- Based on the module to test go to the necessary sections.

### Dynamic Window Module

- Open Terminal in the root folder of the repository.
- Execute: `% cd ./dynamic_window`. All future commands should run from this directory only.
- Create a new Conda Environment `ENCODiT` from the `environment.yml` and activate it.
- Run, `python src/train.py` by setting appropriate window size and dimension of layer 4 of the LeNet Model. The trained model will be saved to the `log` folder.
- From main repository directory run, `python src/inference.py` by setting appropriate window size and choosing model.
