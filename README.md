# Growing Graph Optimization

## Target

The project is used to optimize the growing graph system. Growing graph is an abstract conception, the key idea of this system is the 

## Notice

One important attribute of the comptutation is that it should be fast enough, which means I should ultilize every possible tricks I could
imagine.

## Graph constraints

1. Pose Constraints:
1.1. Uni pose constraints
1.2. Dual pose constraints

2. Geometry Constraints:
2.1. Plane2Plane
2.2. Plane2Surf
2.3. Surf2Surf

## Optimization

Currently, I decided to use intrinsic rotation of ceres to make the optimization, which can be faster.

## Test

A test is needed to validate the performance of graph optimization and remove bugs.