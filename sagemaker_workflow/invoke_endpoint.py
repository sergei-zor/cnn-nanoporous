import argparse
import json
import os
import time

import boto3


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help=".npy structure files")
    parser.add_argument("--endpoint_name", type=str, default="nanoporous-api")
    return vars(parser.parse_args())


def main():
    args = parse_command_line_args()
    client = boto3.client("sagemaker-runtime")
    for path in args["files"]:
        with open(path, "rb") as f:
            body = f.read()
        start = time.time()
        response = client.invoke_endpoint(EndpointName=args["endpoint_name"], ContentType="application/octet-stream", Body=body)
        prediction = json.loads(response["Body"].read())["prediction"]
        # the first request after a while is slower, the endpoint has to start first
        print(f"{os.path.basename(path)}: {prediction:.4f} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
