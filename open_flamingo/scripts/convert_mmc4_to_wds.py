import argparse
import json
import os
import uuid
import zipfile
import pickle

import braceexpand
import webdataset as wds

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--output_dir", type=str)
arg_parser.add_argument(
    "--image_shards",
    type=str,
    help="Pass in a list of shards in the format path_to_shard/shard_{0..23098}_features.pkl",
)
arg_parser.add_argument(
    "--doc_shards",
    type=str,
    help="Pass in a list of shards in the format path_to_shard/shard_{0..23098}.zip",
)
args = arg_parser.parse_args()


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    doc_shards = list(braceexpand.braceexpand(args.doc_shards))
    image_shards = list(braceexpand.braceexpand(args.image_shards))

    assert len(doc_shards) == len(
        image_shards
    ), "Each doc shards must have a corresponding image shard"

    with wds.ShardWriter(args.output_dir + "/%09d.tar", maxcount=1000) as sink:
        for idx in range(len(doc_shards)):

            image_dict = pickle.load(open(image_shards[idx], 'rb'))

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
                            clip_features = image_dict[image_name]

                            # # convert to base64
                            sample_data["image_info"][img_idx][
                                "clip_features"
                            ] = clip_features.tolist()

                        key_str = uuid.uuid4().hex
                        sink.write({"__key__": key_str, "json": sample_data})


if __name__ == "__main__":
    main()