import json
from typing import Dict

# You do not need to add those dependencies to inferencehub/requirements.txt,
# since this file is never used by any of the required methods for deployment on inferencehub.
import requests
from pycognito import Cognito

API_URL: str = "https://api.inferencehub.io/model"


def get_access_token():
    # You can only make requests to our predictions API, if you have an account on InferenceHub
    # This is how you get a valid access token
    u = Cognito(
        user_pool_id="eu-central-1_KrRANFice",
        client_id="2grpmgh0bsn8k1sl867sq6ht7e",
        username="pranav",
    )

    u.authenticate("YOUR PASSWORD")
    return u.id_token


def predict_request(model_name: str, model_domain: str, input: Dict, input_params={}):
    header = {
        "Authorization": f"{get_access_token()}",
    }

    response = requests.post(
        url=f"{API_URL}/{model_domain}/{model_name}/predict",
        files={"input_payload": input, 'input_parameters': json.dumps(input_params)},
        headers=header
    )
    print(f"Response received: {response.json()}")
    return response.json()


if __name__ == "__main__":
    # This is how a pd.Dataframe is formatted when using df.to_json() and
    # how pandas expects it when using pd.read_json(), which is used in inference.py.

    input = {"prod_cycle": {"1": 3}, "base_id": {"1": 10}, "color": {"1": 0}, "start": {"1": 16.304},
             "end": {"1": 71.004}}

    # Input is a python dictionary. To get a valid json you can use:
    json_input = json.dumps(input)
    # or:
    json_input = open("inferencehub/input_sample.json", "rb").read().decode('utf-8')

    # This will time out a few times, when you try it. This is because our server first needs to warm up.
    # After a few times you will get an answer.
    # We are currently working on a faster warm up without time outs.
    predict_request("ai4eu-test", "text", json_input)
