# Automated analysis of the Normi 13 phantom 

(c) Santeri Rytky, Oulu University Hospital, 2024

## Background
Diagnostic X-ray imaging requires maintaining sufficient image quality during the lifetime of the X-ray device.
To facilitate objective and automated quality assurance protocols, 
analysis method was developed for the Normi 13 -phantom.

### Installation
- [Anaconda installation](https://docs.anaconda.com/anaconda/install/) 
```
git clone https://github.com/MIPT-Oulu/natiivi-lv.git
cd natiivi-lv
```
Install the repository requirements
```
pip install -r requirements.txt
```
In case you want to use the analysis codes as a package in another project, install using
```
pip install -e .
```
or
```
pip install git+https://github.com/MIPT-Oulu/natiivi-lv.git
```


## Usage
Populate the input arguments from `analysis_script.py`. Run the script for an example of the analysis. 

## Features

- Automated Phantom edge and orientation detection
- Uniformity analysis from 5 ROIs
- Linearity analysis from 7 copper wedges (0-2.30 mmCu)
- Low-contrast detectability analysis from 6 ROIs (0.8-5.6 %)
- Spatial resolution analysis (1.6-5.0 LP/mm)

## License
This software is distributed under the MIT License.
