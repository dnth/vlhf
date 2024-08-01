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
List dataset on Hugging Face with the keyword "visual"

```python
hf.list_datasets(search="visual")
```

<table>
    <tr>
        <td>&#39;</td>
        <td></td>
        <td>id</td>
        <td>author</td>
        <td>sha</td>
        <td>created_at</td>
        <td>last_modified</td>
        <td>private</td>
        <td>gated</td>
        <td>disabled</td>
        <td>downloads</td>
        <td>likes</td>
        <td>paperswithcode_id</td>
        <td>tags</td>
        <td>\n</td>
        <td>---:</td>
        <td>:-----------------------------------------</td>
        <td>:----------------</td>
        <td>:-----------------------------------------</td>
        <td>:--------------------------</td>
        <td>:--------------------------</td>
        <td>:----------</td>
        <td>:--------</td>
        <td>:-----------</td>
        <td>------------:</td>
        <td>--------:</td>
        <td>:--------------------</td>
        <td>:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
        <td>\n</td>
        <td>0</td>
        <td>visual-layer/oxford-iiit-pet-vl-enriched</td>
        <td>visual-layer</td>
        <td>b4a703833ecb83ba0e96cbc638f7df7ff3f45ba5</td>
        <td>2024-07-04 06:15:06+00:00</td>
        <td>2024-07-29 00:51:33+00:00</td>
        <td>False</td>
        <td>False</td>
        <td>False</td>
        <td>290</td>
        <td>4</td>
        <td></td>
        <td>task_categories:image-classification, task_categories:object-detection, task_categories:visual-question-answering, task_categories:text-to-image, task_categories:image-to-text, language:en, size_categories:1K&lt;n&lt;10K, format:parquet, modality:image, modality:text, library:datasets, library:pandas, library:mlcroissant, library:polars, region:us</td>
        <td>\n</td>
        <td>1</td>
        <td>visual-layer/imagenet-1k-vl-enriched</td>
        <td>visual-layer</td>
        <td>45107c4f5a96e9c2e3d6be3d0a3ca2327b5de3e3</td>
        <td>2024-07-09 08:56:33+00:00</td>
        <td>2024-07-29 00:52:33+00:00</td>
        <td>False</td>
        <td>False</td>
        <td>False</td>
        <td>393</td>
        <td>6</td>
        <td></td>
        <td>task_categories:object-detection, task_categories:image-classification, task_categories:text-to-image, task_categories:image-to-text, task_categories:visual-question-answering, language:en, license:apache-2.0, size_categories:1M&lt;n&lt;10M, format:parquet, modality:image, modality:text, library:datasets, library:dask, library:mlcroissant, library:polars, region:us</td>
        <td>\n</td>
        <td>2</td>
        <td>juletxara/visual-spatial-reasoning</td>
        <td>juletxara</td>
        <td>a07bec7a6b1cbf4b5ca3a68bf744e854982b72bd</td>
        <td>2022-08-11 12:56:58+00:00</td>
        <td>2022-08-11 20:11:21+00:00</td>
        <td>False</td>
        <td>False</td>
        <td>False</td>
        <td>6</td>
        <td>4</td>
        <td></td>
        <td>task_categories:image-classification, annotations_creators:crowdsourced, language_creators:machine-generated, multilinguality:monolingual, source_datasets:original, language:en, license:apache-2.0, size_categories:10K&lt;n&lt;100K, arxiv:2205.00363, arxiv:1908.03557, arxiv:1908.07490, arxiv:2102.03334, region:us</td>
        <td>\n</td>
        <td>3</td>
        <td>albertvillanova/visual-spatial-reasoning</td>
        <td>albertvillanova</td>
        <td>cbe3e224f1ae99617e6188679175ff4a9751a1e3</td>
        <td>2022-12-14 11:31:30+00:00</td>
        <td>2022-12-14 11:55:48+00:00</td>
        <td>False</td>
        <td>False</td>
        <td>False</td>
        <td>0</td>
        <td>4</td>
        <td></td>
        <td>task_categories:image-classification, annotations_creators:crowdsourced, language_creators:machine-generated, multilinguality:monolingual, source_datasets:original, language:en, license:apache-2.0, size_categories:10K&lt;n&lt;100K, arxiv:2205.00363, arxiv:1908.03557, arxiv:1908.07490, arxiv:2102.03334, region:us</td>
        <td>\n</td>
        <td>4</td>
        <td>FastJobs/Visual_Emotional_Analysis</td>
        <td>FastJobs</td>
        <td>31541d6df6c2f5e0b29f0d434327cf02defa83c7</td>
        <td>2023-03-03 06:23:19+00:00</td>
        <td>2023-03-13 06:31:17+00:00</td>
        <td>False</td>
        <td>False</td>
        <td>False</td>
        <td>272</td>
        <td>10</td>
        <td></td>
        <td>task_categories:image-classification, language:en, size_categories:n&lt;1K, format:imagefolder, modality:image, library:datasets, library:mlcroissant, region:us</td>
        <td>\n</td>
        <td>5</td>
        <td>alitourani/moviefeats_visual</td>
        <td>alitourani</td>
        <td>ba9c47d7784a83be0c213eee52bed0ea9139deef</td>
        <td>2024-05-10 17:16:19+00:00</td>
        <td>2024-05-21 20:26:45+00:00</td>
        <td>False</td>
        <td>False</td>
        <td>False</td>
        <td>0</td>
        <td>1</td>
        <td></td>
        <td>task_categories:feature-extraction, task_categories:image-classification, task_categories:video-classification, task_categories:image-feature-extraction, language:en, license:gpl-3.0, arxiv:2309.10461, region:us</td>
        <td>&#39;</td>
    </tr>
</table>

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
