# modal_assurance_criterion:
    1. mac_matrix.py: calculate_mac_matrix of two mode shapes
    2. 模态置信因子 - 知乎.pdf: The source of the calculation formula
# main.py: 
    1. read Abaqus inp file, change the rotation body force. 
    2. write the new inp file. 
    3. run the new inp file. 
    4. read the result file. 
    5. extract modal frequency and mode shape. 
    6. judge which mode shape belongs. 
    7. calculate the distance between calculated mode shapes and prescribed mode shapes. 
    8. write the result to a txt file.
# beam.cae
    1. Abaqus cae file, including the model and the analysis.
