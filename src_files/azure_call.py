import os.path
import time
import numpy as np
import glob
import argparse
from azure.cognitiveservices.vision.contentmoderator import ContentModeratorClient
import azure.cognitiveservices.vision.contentmoderator.models
from msrest.authentication import CognitiveServicesCredentials

class azure_call:
    def __init__(self, input_path: str, subscription_key: str):
        """
        Tests all images in folder against Google Vision NSFW API
        
        Parameters:
            input_path: path of folder containing images to be analyzed. The json file containing the results will also be stored here
            credential_path: path of Google API credential
            
        Returns:
            list of {file, NSFW_likelihood} as well as printing the results and saving it to json file
        
        """
        CONTENT_MODERATOR_ENDPOINT = "https://moderation2.cognitiveservices.azure.com/"
        subscription_key = subscription_key
        client = ContentModeratorClient(
            endpoint=CONTENT_MODERATOR_ENDPOINT,
            credentials=CognitiveServicesCredentials(subscription_key)
        )
        def get_azure_result(img_path):
            with open(img_path, 'rb') as img:
                evl = client.image_moderation.evaluate_file_input(
                    image_stream = img
                )
            return evl.result
        
        input_files = os.listdir(input_path)
        input_files = [file_name for file_name in input_files if (file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'))]
        results = []
        results_path = 'api_results'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        # test files in folder
        for file_path in input_files:
            results.append({
                'file': file_path.split('/')[-1],
                'likelihood': get_azure_result(os.path.join(input_path, file_path))
            })
        print(results)
        # save as json file
        with open(os.path.join(results_path, "azure_results.json"), 'w') as f_out:
            json.dump(results , f_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with input path.")
    parser.add_argument("input_path", help="Path to the input file.")
    parser.add_argument("subscription_key", help="Path to the Google API credential json file.")
    args = parser.parse_args()
    azure_call(args.input_path, args.subscription_key)
