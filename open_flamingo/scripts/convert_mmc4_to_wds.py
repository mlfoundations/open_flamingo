import argparse
import json
import os
import uuid
import zipfile
from PIL import Image
import base64
from io import BytesIO

import braceexpand
import webdataset as wds

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--output_dir",
    type=str,
    help="Pass in the directory where the output shards (as tar files) will be written to.",
)
arg_parser.add_argument(
    "--zip_files",
    type=str,
    help="Pass in a list of MMC4 shards in the format path_to_shard/shard_{0..23098}.zip",
)
arg_parser.add_argument(
    "--image_dir",
    type=str,
    help="Pass in the directory where the images have been downloaded to.",
)
arg_parser.add_argument(
    "--num_files_per_shard",
    type=int,
    default=1000,
)
args = arg_parser.parse_args()


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    doc_shards = list(braceexpand.braceexpand(args.zip_files))

    with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
        for idx in range(len(doc_shards)):
            # Open the ZIP archive and extract the JSON file
            with zipfile.ZipFile(doc_shards[idx], "r") as zip_file:
                # Assumes the JSON file is the first file in the archive
                json_filename = zip_file.namelist()[0]
                with zip_file.open(json_filename, "r") as json_file:
                    for sample_data in json_file:
                        # get image names from json
                        sample_data = json.loads(sample_data)
                        image_info = sample_data["image_info"]
                        image_names = [image["image_name"] for image in image_info]

                        # Add each image to the tar file
                        for img_idx, image_name in enumerate(image_names):
                            try:
                                # load image
                                img = Image.open(
                                    os.path.join(args.image_dir, str(idx), image_name)
                                ).convert("RGB")
                                buffered = BytesIO()
                                img.save(buffered, format="JPEG")
                                img_str = base64.b64encode(buffered.getvalue())

                                # convert to base64
                                sample_data["image_info"][img_idx][
                                    "image_base64"
                                ] = img_str.decode("utf-8")
                            except FileNotFoundError:
                                print(
                                    f"Did not find {image_name} downloaded. This can happen if the url is now 404."
                                )
                            except Exception as e:
                                print(f"Error processing {image_name}: {e}")

                        key_str = uuid.uuid4().hex
                        sink.write({"__key__": key_str, "json": sample_data})

            if (idx + 1) % args.num_files_per_shard == 0:
                sink.next_stream()


if __name__ == "__main__":
    main()
