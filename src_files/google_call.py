from google.cloud import vision
import io
import os 
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
import json

class google_call:    
    def __init__(self, input_path: str, credential_path: str):
        """
        Tests all images in folder against Google Vision NSFW API
        
        Parameters:
            input_path: path of folder containing images to be analyzed. The json file containing the results will also be stored here
            credential_path: path of Google API credential
            
        Returns:
            list of {file, NSFW_likelihood} as well as printing the results and saving it to json file
        
        """
        def implicit():
            from google.cloud import storage
            # If you don't specify credentials when constructing the client, the
            # client library will look for credentials in the environment.
            storage_client = storage.Client()
            # Make an authenticated API request
            buckets = list(storage_client.list_buckets())
            print(buckets)
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
        os.environ['GRPC_DNS_RESOLVER'] = 'native'
        
        def detect_safe_search(path):
            """Detects unsafe features in the file."""
            client = vision.ImageAnnotatorClient()
            with io.open(path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.safe_search_detection(image=image)
            safe = response.safe_search_annotation
            # Names of likelihood from google.cloud.vision.enums
            likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                               'LIKELY', 'VERY_LIKELY')
            if response.error.message:
                raise Exception(
                    '{}\nFor more info on error messages, check: '
                    'https://cloud.google.com/apis/design/errors'.format(
                        response.error.message))
            return safe.adult
        
        # get the list of files
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
                'likelihood': detect_safe_search(os.path.join(input_path, file_path))
            })
        print(results)
        # save as json file
        with open(os.path.join(results_path, "google_results.json"), 'w') as f_out:
            json.dump(results , f_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with input path.")
    parser.add_argument("input_path", help="Path to the input file.")
    parser.add_argument("credential_path", help="Path to the Google API credential.")
    args = parser.parse_args()
    google_call(args.input_path, args.credential_path)
