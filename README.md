# Sketching-lr

This is the code accompanying the paper
"Dobriban, E., & Liu, S. (2019). Asymptotics for sketching in least squares regression. Advances in Neural Information Processing Systems, 32." [(pdf)](https://proceedings.neurips.cc/paper/2019/file/1f36c15d6a3d18d52e8d493bc8187cb9-Paper.pdf)

#### Data sources:
- Million Song Dataset: downloaded from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)
- Flight dataset: from R package [nycflights13](https://cran.r-project.org/web/packages/nycflights13/index.html). We preprocessed the data using the [R script](Experiments/process_flight_data.R). The processed dataset can be found in the folder [datasets](datasets/).

--- 
#### Related works on sketching:
- Sketching for ridge regression: [paper](https://arxiv.org/pdf/1910.02373.pdf)
- Sketching for SVD: [paper](https://ieeexplore.ieee.org/document/9537789), [code](https://github.com/liusf15/sketching-svd)
- Iterative Hessian sketching: [paper](https://proceedings.neurips.cc/paper/2020/file/6e69ebbfad976d4637bb4b39de261bf7-Paper.pdf)


