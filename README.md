# image-distillation
<p align="center">
  <img src="https://github.com/user-attachments/assets/82e8b662-30f5-4111-886a-1109c48bf02d" alt="grid-gif" />
</p>

# Methodology
The `image-distillation` package offers a streamlined approach for selecting representative samples from large image datasets. By first applying Principal Component Analysis, the package reduces the complexity of the data, transforming high-dimensional images into a simpler 2D feature space. Next, KMeans clustering is used to group similar images together. To ensure that the sampled images are both diverse and representative of each group, the package employs a grid-based sampling method. This technique divides the 2D PCA space into various grid shapes, rectangles, parallelograms, triangles, and bricks, and selects samples from each grid cell. This balanced approach helps capture a wide range of image variations within each cluster and across different classes, making the sampling process both efficient and comprehensive.

# Development Process
Starting from the excellent results of the [BLG454E-Learning-from-Data](https://github.com/serdarbicici-visualstudio/BLG454E-Learning-from-Data) project, further research on the [grid-distillation](https://github.com/serdarbicici-visualstudio/grid-distillation) is conducted and as the final result, this Python package is published on [PyPI](https://pypi.org/project/image-distillation/).

# Installation
To install the `image-distillation` package, you can use `pip install`:
```python
!pip install image-distillation
```
You can check the version and package info by:
```python
!pip show image-distillation
```

# Example Usage
Importing the package:
```python
import image_distillation as id
```
Loading an example dataset:
```python
from sklearn import datasets

digits = datasets.load_digits()
data = digits.data
images = digits.images
target = digits.target
```
## Example of `GridSampler`: 
```python
sampler = id.GridSampler(X=images, y=target, n_cluster=10, n_samples=25)
sampler.cluster_images()
sampler.plot_clusters()
sampler.plot_clusters_with_grids()
selected_samples = sampler.get_selected_samples()
sampler.plot_selected_samples_on_clusters(selected_samples)
```
## Example of `ParallelogramSampler`: 
```python
sampler = id.ParallelogramSampler(X=images, y=target, n_cluster=10, n_samples=25)
sampler.cluster_images()
sampler.plot_clusters()
sampler.plot_clusters_with_parallelograms()
selected_samples = sampler.get_selected_samples()
sampler.plot_selected_samples_on_clusters(selected_samples)
```
## Example of `TriangularSampler`: 
```python
sampler = id.TriangularSampler(X=images, y=target, n_cluster=10, n_samples=25)
sampler.cluster_images()
sampler.plot_clusters()
serdar.plot_clusters_with_triangles()
selected_samples = sampler.get_selected_samples()
sampler.plot_selected_samples_on_clusters(selected_samples)
```
## Example of `BrickSampler`: 
```python
sampler = id.BrickSampler(X=images, y=target, n_cluster=10, n_samples=25)
sampler.cluster_images()
sampler.plot_clusters()
sampler.plot_clusters_with_bricks()
selected_samples = sampler.get_selected_samples()
sampler.plot_selected_samples_on_clusters(selected_samples)
```
## Example Outputs
<div style="display: flex; justify-content: center;">
    <div style="display: flex; flex-direction: column;">
        <div style="display: flex; justify-content: center;">
            <img src="https://github.com/user-attachments/assets/6c498d53-2c3d-4175-9c93-45f7d638360a" alt="Image 1" style="width: 45%; margin: 5px;">
            <img src="https://github.com/user-attachments/assets/1a098135-591c-4190-bf35-09f290244600" alt="Image 2" style="width: 45%; margin: 5px;">
        </div>
        <div style="display: flex; justify-content: center;">
            <img src="https://github.com/user-attachments/assets/5a779d32-b80f-4077-846f-9d4797c30e79" alt="Image 3" style="width: 45%; margin: 5px;">
            <img src="https://github.com/user-attachments/assets/edaa5523-bd65-4e65-a972-af4d4458a82b" alt="Image 4" style="width: 45%; margin: 5px;">
        </div>
    </div>
</div>

Plotting with grids is not perfect in released versions. However, they can be used to visualize the process.


for questions and comments, you can reach out to [Serdar Biçici](https://github.com/serdarbicici-visualstudio)
