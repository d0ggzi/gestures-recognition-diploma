import Augmentor

gesture = "peace"
p = Augmentor.Pipeline(source_directory=f"./gestures/originals/{gesture}",
                       output_directory=f"../../augmentated/{gesture}")

p.crop_random(probability=0.5, percentage_area=0.9)
p.random_distortion(probability=0.1, grid_width=2, grid_height=2, magnitude=2)
p.flip_left_right(probability=0.5)
p.sample(100)
