# fish-detecting
### Dataset can be found at:
#### https://www.fisheries.noaa.gov/west-coast/science-data/labeled-fishes-wild
cite: Cutter, G.; Stierhoff, K.; Zeng, J. (2015) "Automated detection of rockfish in unconstrained underwater videos using Haar cascades and a new image dataset: labeled fishes in the wild," IEEE Winter Conference on Applications of Computer Vision Workshops, pp. 57-62.

## Goals:
1. The first goal was to track the fish in the ROV video and this was successful with a HAAR model and approximately 40 stages of training.
2. The second goal of the project was to classify the fish being tracked.
- So far the main issue is defining which fish are of what species.
- The training images for the classification model can be found in the dataset or the fish_on_static folder.
- The current solution uses K-Means clustering to sort the fish into relative species.
