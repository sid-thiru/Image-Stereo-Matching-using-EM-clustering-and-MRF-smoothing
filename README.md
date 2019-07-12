# STEREO MATCHING

If there are multiple cameras taking the same photograph from different positions, the depth of the objects can be recovered. 
The aim of this script is to extract the depth map of a given set of stereo images.

## Overview of the steps involved
* im0.ppm (left) and im8.ppm (right) are the pictures taken by two different camera positions
* The disparity matrix between the two images is computed
* Expectation Maximization is used to cluster the disparity values in this matrix. The clustered disparity matrix gives us the depth map for the given stereo image pair.
* The implementation is extended by MRF smoothing priors using an eight neighborhood system. This is done by Gibbs sampling.