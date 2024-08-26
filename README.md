
# Overview
This repository contains source code of Theia, benchmark used for empirical evaluation, patches applied to the buggy and correct models and posts used for mapping dataset characteristics with structural bugs.  
```
.
├── SOF                                    # Contains 40 buggy programs obtained from Stack Overflow 
├── Theia                                  # Source code of Theia
    ├── Theia_keras.py                     # Contains callback for Keras framework
    ├── Theia_torch.py                     # Contains callback for PyTorch framework
    ├── Instructions.txt                   # Intructions for using Theia
    └── requirements.txt                   # Dependency and Python virutal environment information
├── buggy_models_with_patch                # Contains original buggy programs and patch/fix suggested by Theia and NeuraLint
├── correct_models_with_patch              # Contains correct programs and patch/fix suggested by Theia and NeuraLint
└── 105_mapping_posts.xlsx                 # Contains posts used for mapping dataset characteristics with structural bugs
```
# SOF
The 40 buggy programs used for empiricial evaluation are stored under the directory [SOF](SOF). Each buggy program is stored in a file named after the StackOverflow post handle. 

# Theia
The directory [Theia](Theia) contains the source code of the callbacks for Keras framework [callbacks](Theia_keras.py) and callbacks for PyTorch framework [callbacks](Theia_torch.py). To run Theia, one needs to create a virtual environment. The instructions for creating virtual environment and how to use KUnit for mock testing are provided in [Instructions.txt](KUnit/Instructions.txt). The motivation example is provided as a reference example. Follow the instructions to reproduce the results.


# Buggy Models with Patch
The 40 buggy programs along with the patches applied manually following the line numbers and fix suggestions provided by Theia and NeuraLint are under the directory [buggy_models_with_patch](buggy_models_with_patch). Each buggy program along with its patches are stored in a folder named after the StackOverflow post handle. If Theia or NeuraLint do not report any bug then the program after fix is not included in the folder. These programs are used to report the performance of the models before/after applying patches in Section 5.3.3 of the paper.

# Correct Models with Patch
The 40 correct programs obtained after applying the patches used as groud truth along with the patches applied manually following the line numbers and fix suggestions provided by Theia and NeuraLint are under the directory [correct_models_with_patch](correct_models_with_patch). Each correct program along with its patches are stored in a folder named after the StackOverflow post handle. If Theia or NeuraLint do not report any bug then the program after fix is not included in the folder. These programs are used to report the performance of the models before/after applying patches in Section 5.3.4 of the paper.

# Mapping between Dataset Characteristics and Structural Bugs
The 105 Stack Overflow posts used for mapping between dataset characteristics with structural bugs in Section 3.2 of the paper are provided in [105_mapping_posts.xlsx](105_mapping_posts.xlsx).
