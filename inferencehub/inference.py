import pandas as pd


def preprocess_function(input_payload: object) -> object:
    ## compute duration and one-hot-encode `colo`
    df_ml = input_payload.copy()
    df_ml['duration'] = df_ml['end'] - df_ml['start']
    df_ml = pd.concat([
            df_ml
            , pd.get_dummies(df_ml['color'], prefix='color')]
        ,  axis=1)
    
    # make all cycles start at time 0
    df_ml['start'] = (df_ml['start']
                      - df_ml.groupby('prod_cycle')['start'].transform('min')
                     )
    
    return df_ml
    


def postprocess_function(output: int) -> int:
    # postprocessing
    return output
