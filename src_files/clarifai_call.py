import os
import json
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_pb2, status_code_pb2
import argparse

class clarifai_call:
    

    def __init__(self, input_path: str, api_key: str, model_id: str = 'nsfw-recognition'):
        '''
        Tests all images in folder against Clarifai NSFW API
        
        Parameters:
            input_path: path of folder containing images to be analyzed. The json file containing the results will also be stored here
            api_key: ClarifAI API key
            model_id: the model ID of content moderation
        Returns:
            list of {file, NSFW_likelihood} as well as printing the results and saving it to json file
        '''
        ##############################################################################
        ## Initialize client
        ##     - This initializes the gRPC based client to communicate with the 
        ##       Clarifai platform. 
        ##############################################################################
        ## Import in the Clarifai gRPC based objects needed

        ## Construct the communications channel and the object stub to call requests on.
        # Note: You can also use a secure (encrypted) ClarifaiChannel.get_grpc_channel() however
        # it is currently not possible to use it with the latest gRPC version
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        ################################################################################
        ## Set up Personal Access Token and Access information
        ##     - This will be used by every Clarifai API call 
        ################################################################################
        ## Specify the Authorization key.  This should be changed to your Personal Access Token.
        ## Example: metadata = (('authorization', 'Key 123457612345678'),) 
        metadata = (('authorization', api_key),)

        def get_nsfw(path: str, model_id: str) -> dict:
            '''
            Parameters:
                path: path of folder of images for analysis
                model_id: id of the Clarifai model
            Returns:
                
            '''
            # load image as 64bit byte file, if sending as json, must be encoded in base64
            with open(path, "rb") as image:
                f = image.read()
                byte = bytes(f)
            # make request
            request = service_pb2.PostModelOutputsRequest(
            model_id = model_id,
            inputs=[
              resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(base64 = byte)))
            ])
            response = stub.PostModelOutputs(request, metadata=metadata)
            # error message if the API call returns error
            if response.status.code != status_code_pb2.SUCCESS:
                print("There was an error with your request!")
                print("\tCode: {}".format(response.outputs[0].status.code))
                print("\tDescription: {}".format(response.outputs[0].status.description))
                print("\tDetails: {}".format(response.outputs[0].status.details))
                raise Exception("Request failed, status code: " + str(response.status.code))
            # return the likelihood of NSFW
            output = response.outputs[0].data
            if output.concepts[0].name == 'nsfw':
                return output.concepts[0].value
            else:
                return output.concepts[1].value
        
        
        
        # get the list of files
        input_files = os.listdir(input_path)
        input_files = [file_name for file_name in input_files if (file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'))]
        results = dict()
        results_path = 'api_results'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        # test files in folder
        for file_path in input_files:
            nsfw_score = get_nsfw(os.path.join(input_path, file_path), model_id=model_id)
            results[file_path] = nsfw_score
        print(results)
        # save as json file
        with open(os.path.join(results_path, "clarifai_results.json"), 'w') as f_out:
            json.dump(results , f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with input and API key.")
    parser.add_argument("input_path", help="Path to the input file.")
    parser.add_argument("api_key", help="key to Clarifai API")
    args = parser.parse_args()
    clarifai_call(args.input_path, args.api_key)
