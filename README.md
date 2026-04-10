# Multispectral Image Preprocessing Pipeline

This repository includes a script, `metashape.py`, that automates Agisoft Metashape processing for multispectral AOI subsets.

The script does all of the following for each AOI/ratio combination:

1. Creates or opens a Metashape project.
2. Adds selected survey photos and panel photos.
3. Aligns cameras.
4. Runs reflectance calibration using panel + sun sensor.
5. Builds depth maps, dense point cloud, DEM, and orthomosaic.
6. Saves the Metashape project and logs processing times.

## 1. Requirements

### Software

- Agisoft Metashape Professional (required for the `Metashape` Python API).
- Python 3.10 or newer.
- Conda.

### Python packages used by `metashape.py`

- pandas
- geopandas
- numpy
- pyproj

The script also imports helper functions from `aoi_filtering.py`, which depends on:

- shapely
- micasense imageprocessing package
- tqdm

## 2. Environment Setup

Create and activate a conda environment:

```
conda create -n micasense python=3.10 -y
conda activate micasense
```

Install core geospatial dependencies:

```
pip install tqdm numpy pandas pyproj shapely geopandas
```

Install MicaSense imageprocessing package:

```
conda install conda-forge::gdal -y
pip install git+https://github.com/micasense/imageprocessing.git
```

Install ExifTool and set `exiftoolpath` (required by MicaSense metadata parsing):

1. Download from https://exiftool.org/.
2. Place it in a stable location, for example `C:\exiftool\exiftool.exe`.
3. Add user environment variable:
     - Name: `exiftoolpath`
     - Value: `C:\exiftool\exiftool.exe`

Install Metashape Python API:

1. Download from https://download.agisoft.com/metashape-2.3.0-cp39.cp310.cp311.cp312.cp313-none-win_amd64.whl
2. pip install metashape-2.3.0-cp39.cp310.cp311.cp312.cp313-none-win_amd64.whl

## 3. Project Data Layout

`metashape.py` expects this structure under `Data/<experiment_name>/`.

```text
Data/
    aoi.csv
    <experiment_name>/
        imageSet.json
        Images/
            <capture_id>_1.tif
            <capture_id>_2.tif
            <capture_id>_3.tif
            <capture_id>_4.tif
            <capture_id>_5.tif
        Panel/
            IMG_0000_1.tif
            IMG_0000_2.tif
            IMG_0000_3.tif
            IMG_0000_4.tif
            IMG_0000_5.tif
        Metashape/
            <experiment_name>.psx   # created automatically if missing
```

### Required file details

#### `aoi.csv`

- Must contain at least columns `Latitude` and `Longitude` in EPSG:4326.
- Each row is a potential AOI center.
- `aoi_id` in the script is the row index used for processing.

#### `imageSet.json`

- GeoJSON of capture footprints/points.
- Must include an `image_name` attribute used to build survey image file names.

#### Image naming convention

- Survey images are resolved as `<image_name>_<band>.tif` for bands 1..5.
- Panel images are resolved as `IMG_0000_<band>.tif` for bands 1..5.

If your file names differ, update the list comprehensions in `metashape.py` accordingly.


## 4. Configure `metashape.py`

Open `metashape.py` and set these values in the `__main__` block:

- `parent_folder` (default `Data`)
- `exp` (experiment folder name)
- `aoi_id` (row index in `aoi.csv`)
- `original_crs` (usually `EPSG:4326`)
- `target_crs` (UTM or project CRS, for example `EPSG:32616`)

The script currently loops over overlap expansion values:

- `ratio` from 0.05 to 0.50 in steps of 0.05.

For each ratio it creates a new Metashape chunk labeled:

- `AOI_<aoi_id>_ratio_<ratio>`

## 5. Run the Script

To run the python script:

```
python metashape.py
```


## 7. Outputs

After completion:

- Metashape project: `Data/<exp>/Metashape/<exp>.psx`
- Timing summary CSV: `Data/<exp>/Metashape/processing_results.csv`
- Multiple chunks in the `.psx`, one per ratio value.

`processing_results.csv` contains:

- `ratio`
- `num_captures`
- `process_time` (seconds)


