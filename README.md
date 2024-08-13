# Scene Text Detection
The goal of this project is to implement a scene text detector that can detect semantic phrases.

## First Iteration
### Functionality
1. Text in the scene are detexted using EAST text detector.
2. Cluster words that are close by together using K-Means Clustering Algorithm
3. Sort words the order of words similar to human reading conversions for English.

### Problems
1. Need to know the number of clusters beforehand.
2. Low performance bounding box detection of text
3. Lack of Layout context like fonts and font size.

### Future Work
1. Use cluster-agnostic clustering algorithm
2. Use more accurate text detectors
3. Integrate background context, Font and Layout information

### Tests
![test2](https://github.com/user-attachments/assets/bde76b81-5dae-4363-8b89-17434fb67ecb)
![test1](https://github.com/user-attachments/assets/edd190c4-6628-4a73-ab40-cec94d3c5280)

