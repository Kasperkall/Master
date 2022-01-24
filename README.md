# Specialization project: Weldline laser-reflection removal

This project compares the use of UNET and a simple CNN to remove laser-reflection. The models are tweaked with different combination of hyperparameters such as batch size, optimizers, learning rate and loss functions. The configurations are easily changed in the config.yaml file in the "config" folder. The dataset containing the images and ground truths of the welds was generated by Ola Alstad during his master thesis. In short, he used Blender, a 3D computer graphics program, and a ray tracing engine named LuxCoreRender to produce images of the welds. To use these images specific dataloaders were also made for this project.
