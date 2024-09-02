
# Overview
This repository contains the source code of Theia, a benchmark used for empirical evaluation, patches applied to the buggy and normal models, posts used for mapping dataset characteristics with structural bugs and details of manual labeling process. 
```
.
├── Benchmark                              # Contains 40 buggy programs obtained from Stack Overflow 
├── Theia                                  # Source code of Theia
    ├── Theia_keras.py                     # Contains callback for Keras framework
    ├── Theia_torch.py                     # Contains callback for PyTorch framework
    ├── Instructions.txt                   # Intructions for using Theia
    └── requirements.txt                   # Dependency and Python virutal environment information
├── buggy_models_with_patch                # Contains original buggy programs and patch/fix suggested by Theia and NeuraLint
├── correct_models_with_patch              # Contains correct programs and patch/fix suggested by Theia and NeuraLint
├── 105_mapping_posts.xlsx                 # Contains posts used for mapping dataset characteristics with structural bugs
└── Manual_Labeling_for_Mapping.xlsx       # Contains the labels from the first round and the final labels.
```
# Benchmark
The 40 buggy programs used for empiricial evaluation are stored under the directory [Benchmark](https://github.com/anoau/Theia/tree/main/Benchmark). Each buggy program is stored in a file named after the Stack Overflow post handle. 

# Theia
The directory [Theia](https://github.com/anoau/Theia/tree/main/Theia) contains the source code of the callbacks for Keras framework [callbacks](https://github.com/anoau/Theia/blob/main/Theia/Theia_keras.py) and callbacks for PyTorch framework [callbacks](https://github.com/anoau/Theia/blob/main/Theia/Theia_torch.py). To run Theia, one needs to create a virtual environment. The instructions for creating virtual environment and how to use KUnit for mock testing are provided in [Instructions.txt](https://github.com/anoau/Theia/blob/main/Theia/Instructions.txt). The motivation example is provided as a reference example. Follow the instructions to reproduce the results.


# Buggy Models with Patch
The 40 buggy programs, along with the patches applied manually following the line numbers and fix suggestions provided by Theia and NeuraLint, are under the directory [buggy_models_with_patch](https://github.com/anoau/Theia/tree/main/buggy_models_with_patch). Each buggy program, along with its patches, is stored in a folder named after the Stack Overflow post handle. If Theia or NeuraLint does not report any bug, then the program after the fix is not included in the folder. These programs are used to report the performance of the models before/after applying patches in Section 5.3.3 of the paper.

# Normal Models
The 40 normal programs obtained after applying the patches used as ground truth, along with the patches applied manually following the line numbers and fix suggestions provided by Theia and NeuraLint, are under the directory  [normal_models_with_patch](https://github.com/anoau/Theia/tree/main/normal_models_with_patch). Each normal program, along with its patches, is stored in a folder named after the Stack Overflow post handle. If Theia or NeuraLint does not report any bug, then the program after the fix is not included in the folder. These programs are used to report the performance of the models before/after applying patches in Section 5.3.4 of the paper.

# Mapping between Dataset Characteristics and Structural Bugs
The 105 Stack Overflow posts with non-crash bugs obtained from dataset of bugs released by Humbatova et al. [62] used for mapping dataset characteristics with structural bugs are provided in [105_mapping_posts.xlsx](https://github.com/anoau/Theia/blob/main/105_mapping_posts.xlsx).

#  Manual Labeling
All the files associated with our manual labeling process are provided in  [Manual_Labeling_for_Mapping.xlsx](https://github.com/anoau/Theia/blob/main/Manual_Labeling_for_Mapping.xlsx).
