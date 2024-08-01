# VLHF: Visual Layer - Hugging Face Integration

VLHF (Visual Layer - Hugging Face) is a Python package that provides a seamless interface for transferring datasets between Visual Layer and Hugging Face.

## Features

- Upload datasets from Visual Layer to Hugging Face
- Download datasets from Hugging Face to Visual Layer
- Abstract away complexities of data transfer between platforms

## Installation

### Prerequisites

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
vl = VisualLayer(VL_USER_ID, VL_ENV)
```
List dataset on Hugging Face

```python
hf.list_datasets(search="visual")
```

Download a dataset from Hugging Face

```python
hf.download_dataset(dataset_id="lewtun/dog_food", image_key="image", label_key="label")
```
Parameters:
+ dataset_id: The dataset ID on Hugging Face datasets.
+ image_key: The column name in the dataset that contains PIL images.
+ label_key: The column name containing image classification labels.


> [!NOTE]  
> Not all datasets use "image" and "label" as their column names. Adjust these parameters based on the specific dataset structure.

Upload to Visual Layer

```python
hf.to_vl(vl_session=vl)
```
