# Machine_Learning_Algorithms_Applied_to_Mice_Protein_Data

Higuera et al. (2015) applied machine learning techniques to control and Down Syndrome (DS) mice models to determine which of the 77 proteins of interest played a more important role in learning and memory deficits associated with DS.

This repository comes with:
  Data folder:
    - Data_Cortex_Nuclear(corrected).xlsx: contains the data set.
    - Readme_for_Data_Cortex_Nuclear.txt: contains a brief overview of the data.
  - main.py: contains the python script that runs the analysis.
  - CLC vs CNLC Figure.png: which is a sample figure produced by the running the analysis comparing c-CS-s (control mice in a fear learning condition) and c-SC-s (control mice in a non-learning condition). Additional conditions can be used to run the analysis. The mice code can be found in the reference material as well as the files in the data folder.
  
The program first applies PCA to the data for dimension reduction (from 77 features/proteins to 2). Then, the Gaussian Naive Bayes classifier (GNBC) is applied to the data. On the figure the average performance of 100 runs of GNBC is shown along with the decision boundary.

The interesting finding from the sample PCA output was that 2 (i.e. ERK and pERK) of the top 5 (NR2A, ERK, pELK, Bcatenin, and pERK) proteins (out of a total of 77) that explained the most variability in the data have been reported to be related to learning and memory in the MAPK signaling pathway [2]. The GNBC was able to correctly classify which condition the mice belonged to with approximately 65% accuracy.

Additionally, a support vector classifier (SVC) was applied to the data to discriminate between two mice groups. The performance of the SVC showed to be generally better than GNBC approximating 80% accuracy as opposed to 70% accuracy. NOTE: these numbers are not reported based on detailed quantification. They are reported based on a qualitative assessment from the trials I have run.

References:

[1] Higuera, C., Gardiner, K. J., & Cios, K. J. (2015). Self-organizing feature maps identify proteins critical to learning in a mouse model of down syndrome. PloS one, 10(6), e0129126.

[2] Sweatt, J. D. (2001). The neuronal MAP kinase cascade: a biochemical signal integration system subserving synaptic plasticity and memory. Journal of neurochemistry, 76(1), 1-10.
