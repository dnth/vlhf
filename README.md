# VLHF: Visual Layer - Hugging Face Integration

![image](assets/vlhf.jpg)

VLHF (Visual Layer - Hugging Face) is a Python package that provides a seamless interface for transferring datasets between Visual Layer and Hugging Face.

## Features

- Download/Upload datasets from Hugging Face to local storage.
- Download/Upload datasets from local storage to Visual Layer.
- Search for datasets on Hugging Face.

> [!NOTE] 
> Currently only works for image classification datasets. Datasets with bounding boxes are currently work in progress.


## Installation

### Prerequisites
Python 3.10 or higher is required.

Before installing VLHF, you need to install the vl-research package:

```bash
git clone https://github.com/visual-layer/vl-research
cd vl-research
pip install -e .
```

### Install vlhf
To install the vlhf package, run:

```
pip install -e .
```

## Usage

Authentication

```python
from vlhf.hugging_face import HuggingFace
from vlhf.visual_layer import VisualLayer

hf = HuggingFace(HF_TOKEN)
vl = VisualLayer(VL_USER_ID, VL_ENV, VL_PG_URI)
```
List dataset on Hugging Face with the search term "visual"

```python
hf.list_datasets(search="visual")
```

<table>
    <tr>
        <th>id</th>
        <td>author</td>
        <td>sha</td>
        <td>created_at</td>
        <td>private</td>
        <td>downloads</td>
        <td>likes</td>
        <td>tags</td>
    </tr>
    <tr>
        <th>0</th>
        <td>visual-layer/oxford-iiit-pet-vl-enriched</td>
        <td>b4a70383...</td>
        <td>2024-07-04 06:15:06</td>
        <td>False</td>
        <td>290</td>
        <td>4</td>
        <td>task_categories:image-classification, task_cat...</td>
    </tr>
    <tr>
        <th>1</th>
        <td>visual-layer/imagenet-1k-vl-enriched</td>
        <td>45107c4f...</td>
        <td>2024-07-09 08:56:33</td>
        <td>False</td>
        <td>393</td>
        <td>6</td>
        <td>task_categories:object-detection, task_categor...</td>
    </tr>
    <tr>
        <th>2</th>
        <td>juletxara/visual-spatial-reasoning</td>
        <td>a07bec7a...</td>
        <td>2022-08-11 12:56:58</td>
        <td>False</td>
        <td>6</td>
        <td>4</td>
        <td>task_categories:image-classification, annotati...</td>
    </tr>
    <tr>
        <th>3</th>
        <td>albertvillanova/visual-spatial-reasoning</td>
        <td>cbe3e224...</td>
        <td>2022-12-14 11:31:30</td>
        <td>False</td>
        <td>0</td>
        <td>4</td>
        <td>task_categories:image-classification, annotati...</td>
    </tr>
    <tr>
        <th>4</th>
        <td>FastJobs/Visual_Emotional_Analysis</td>
        <td>31541d6d...</td>
        <td>2023-03-03 06:23:19</td>
        <td>False</td>
        <td>272</td>
        <td>10</td>
        <td>task_categories:image-classification, language...</td>
    </tr>
    <tr>
        <th>5</th>
        <td>alitourani/moviefeats_visual</td>
        <td>ba9c47d7...</td>
        <td>2024-05-10 17:16:19</td>
        <td>False</td>
        <td>0</td>
        <td>1</td>
        <td>task_categories:feature-extraction, task_categ...</td>
    </tr>
</table>

### From HF to VL

Download a dataset from Hugging Face

```python
hf.download_dataset(dataset_id="lewtun/dog_food", image_key="image", label_key="label")
```
Parameters:
+ `dataset_id`: The dataset ID on Hugging Face datasets.
+ `image_key`: The column name in the dataset that contains PIL images.
+ `label_key`: The column name containing image classification labels.


> [!NOTE]  
> Not all datasets use `"image"` and `"label"` as their column names. Adjust these parameters based on the specific dataset structure.

Upload to Visual Layer

```python
hf.to_vl(vl_session=vl)
```

Parameters:
+ `vl_session`: The authenticated Visual Layer session object.


### From VL to HF
Get dataset from Visual Layer

```python
dataset_id = "124aa35a-4fd3-11ef-ab8c-7e1db6b41710"
vl.get_dataset(dataset_id=dataset_id) # returns a polars DataFrame
```

<table border="1">
  <thead>
    <tr>
      <th>image_uri</th>
      <th>image_label</th>
      <th>image_issues</th>
      <th>object_labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>https://d2iycf…</td>
      <td>dog</td>
      <td>null</td>
      <td>null</td>
    </tr>
    <tr>
      <td>https://d2iycf…</td>
      <td>dog</td>
      <td>null</td>
      <td>null</td>
    </tr>
    <tr>
      <td>https://d2iycf…</td>
      <td>muffin</td>
      <td>null</td>
      <td>null</td>
    </tr>
    <tr>
      <td>https://d2iycf…</td>
      <td>chicken</td>
      <td>null</td>
      <td>null</td>
    </tr>
    <tr>
      <td>https://d2iycf…</td>
      <td>dog</td>
      <td>null</td>
      <td>null</td>
    </tr>
    <tr>
      <td>https://d2iycf…</td>
      <td>muffin</td>
      <td>null</td>
      <td>null</td>
    </tr>
    <tr>
      <td>https://d2iycf…</td>
      <td>muffin</td>
      <td>null</td>
      <td>null</td>
    </tr>
    <tr>
      <td>https://d2iycf…</td>
      <td>dog</td>
      <td>null</td>
      <td>null</td>
    </tr>
    <tr>
      <td>https://d2iycf…</td>
      <td>dog</td>
      <td>null</td>
      <td>null</td>
    </tr>
  </tbody>
</table>

Upload to Hugging Face

```python
hf_repo_id = "dnth/dog_food-vl-enriched"
vl.to_hf(hf_session=hf, hf_repo_id)
```

Parameters:
+ `hf_session`: The authenticated Hugging Face session object.


## Development

Run pre-commit to lint and format the code:

```bash
pre-commit run --all-files
```

Run mypy to check for type errors:

```bash
mypy src/
```