# Pre-processing Object Files

Pre-processing is used to **centre the origin** of the reference-frame related to the object to the
centre of the object mass, i.e. to centre the object in its own reference frame. It is also used to
**tune the number of faces** of which the model is composed. A lower number of faces reduces
the computational cost with rasterising the object, but also reduces the object detail.

The software used to modify the object is MeshLab http://www.meshlab.net/

`sudo apt install meshlab`

To centre an object model that has been loaded into MeshLab:
  - From the menu bar: Filters → Normals, Curvatures and Orientation → Transform: Translate, Center, set Origin
  - Transformation = Center on BBox → Apply → Close
  - From the menu bar select: Draw XYZ Axis in world coordinates, to check the centering has been performed correctly.
  
To reduce the number of faces comprising a model that has been loaded into MeshLab:
  - From the menu bar: Filters → Cleaning and Repairing → Remove Duplicate Faces
  - From the menu bar: Filters → Cleaning and Repairing → Remove Duplicate Vertices
  - From the menu bar: Filters → Remeshing, Simplification and Reconstruction → Simplification: Quadratic Edge Collapse Decimation (with Texture)
  - Input the new target number of faces, texture weight = 10, planar simplification = YES → Apply
