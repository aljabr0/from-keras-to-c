# from-keras-to-c
This small demo project is about deploying deep learning models on embedded platforms.
The techniques exposed here have been particularly useful to me in the deployment of deep learning models in industrial applications.

We start with a simple example model, trained with Tensorflow + Keras. At the end we freeze the model and export a GraphDef that
can be loaded and executed through the Tensorflow C API. The training and export are coded into the file train_and_export.py whereas
the inference is coded into the file model_run.cpp.
