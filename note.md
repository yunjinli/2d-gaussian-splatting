```
pixi run python render_mesh_multiview.py meshes/table.glb --output data/table
pixi run python train.py -s data/table --init_mesh meshes/table.glb --white_background --lambda_normal 0.05 --lambda_dist 0.0
```