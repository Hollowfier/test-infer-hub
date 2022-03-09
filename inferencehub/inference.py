from io import BytesIO

import pandas as pd
import tensorflow as tf


def preprocess_function(input_payload: BytesIO) -> tf.Tensor:
    ## compute duration and one-hot-encode `colo`

    # We are currently updating how to input the data into our API.
    # So this differs from the current version of our documentation at https://docs.inferencehub.io/,
    # but we will update it soon.
    # From this version forward all API requests will be passed as BytesIO objects to the preprocessing function.
    # The line below thus converts the given BytesIO object to a string
    json_string = input_payload.read().decode("utf-8")

    # Since you are using pandas below, the input needs to be converted to a pandas Dataframe.
    # How you do this conversion depends on how you make your reqests.
    # The line below uses the pandas read_json-method.
    # You can see a corresponding example request in example_predict_request.py
    df_ml = pd.read_json(json_string)

    ####################
    # Your Preprocessing
    df_ml['duration'] = df_ml['end'] - df_ml['start']
    df_ml = pd.concat([
            df_ml
            , pd.get_dummies(df_ml['color'], prefix='color')]
        ,  axis=1)

    # make all cycles start at time 0
    df_ml['start'] = (df_ml['start']
                      - df_ml.groupby('prod_cycle')['start'].transform('min')
                     )
    ####################

    # The return value of this function will be directly passed to the predict-method of your keras model.
    # So the return value should be a tensor of shape (1, 1, 4), since this is what your model expects.
    # The pandas Dataframe has currently 7 entries, so 3 entries need to be dropped.
    # I don't know what entries your model expects as inputs, so I just chose the first 4.
    # If the chosen entries are the wrong ones, feel free to fix it and contact us (support@dmesh.io),
    # so we can redeploy your model.
    # A convenient way for redeployment is currently in development, so for now we can do it manually for you.
    df_ml.drop(df_ml.columns[[0, 1, 2, 4, 5]], axis = 1, inplace = True)

    # Tensorflow automatically converts numpy arrays to tf.Tensors, so it is sufficient to return a numpy array
    df_ml = df_ml.to_numpy()

    # The numpy array is currently of shape (1, 4), your model expects a shape of (1, 1, 4),
    # so the line below adds one dimension.
    df_ml = df_ml[None]

    return df_ml


def postprocess_function(output: tf.Tensor) -> float:
    # postprocessing

    # Your model outputs a Tensor of shape (1, 1, 1). Which means the output is one float packed inside three arrays.
    # For convenience the line below unpacks the float and returns it,
    # so that the response from our API will be a float and not a tensor
    return output.numpy()[0][0][0]
