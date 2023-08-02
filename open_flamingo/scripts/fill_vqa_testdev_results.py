"""
Helper scripts to prepare a vqa test-dev evaluation for EvalAI submission.
Note: EvalAI requires VQAv2 submissions to have predictions for all the questions in the test2015 set, not just the test-dev set.
Given a json with a subset of the vqa questions, fill in the rest of the questions with an empty string as the model prediction.
"""
import json
import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
    )
)
from eval.vqa_metric import VQAEval

postprocessor = VQAEval(None, None)


def fill_vizwiz_test_json(
    input_path,
    output_path,
    vqa_test_questions_json_path,
):
    # read the input json and build a set with all question_ids
    with open(input_path, "r") as f:
        input_json = json.load(f)

    # postprocess answers
    question_id_to_answer = {}
    for q in input_json:
        resAns = q["answer"]
        resAns = resAns.replace("\n", " ")
        resAns = resAns.replace("\t", " ")
        resAns = resAns.strip()
        resAns = postprocessor.processPunctuation(resAns)
        resAns = postprocessor.processDigitArticle(resAns)
        question_id_to_answer[q["question_id"]] = resAns

    # read the vqa test json to get all the qustion_ids that need to be filled
    with open(vqa_test_questions_json_path, "r") as f:
        vqa_test_json = json.load(f)
    vqa_test_json = vqa_test_json["questions"]

    # if the question_id is not in the set, add it to the copy of the input json with an empty string as the answer
    output_json = []
    for q in vqa_test_json:
        output_json.append(
            {
                "image": q["image_id"],
                "answer": question_id_to_answer.get(q["question_id"], ""),
            }
        )

    # write the json to the output path
    with open(output_path, "w") as f:
        json.dump(output_json, f)


def fill_vqav2_test_json(
    input_path,
    output_path,
    vqa_test_questions_json_path,
):
    # read the input json and build a set with all question_ids
    with open(input_path, "r") as f:
        input_json = json.load(f)
    question_ids = set()
    for q in input_json:
        question_ids.add(q["question_id"])

    # make a copy of the input json
    output_json = []
    for q in input_json:
        resAns = q["answer"]
        resAns = resAns.replace("\n", " ")
        resAns = resAns.replace("\t", " ")
        resAns = resAns.strip()
        resAns = postprocessor.processPunctuation(resAns)
        resAns = postprocessor.processDigitArticle(resAns)
        q["answer"] = resAns
        output_json.append(q)

    # read the vqa test json to get all the qustion_ids that need to be filled
    with open(vqa_test_questions_json_path, "r") as f:
        vqa_test_json = json.load(f)
    vqa_test_json = vqa_test_json["questions"]

    # if the question_id is not in the set, add it to the copy of the input json with an empty string as the answer
    for q in vqa_test_json:
        if q["question_id"] not in question_ids:
            output_json.append(
                {
                    "question_id": q["question_id"],
                    "answer": "",
                }
            )

    # write the json to the output path
    with open(output_path, "w") as f:
        json.dump(output_json, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["vqav2", "vizwiz"],
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the json file with the subset of the vqa test-dev questions.",
    )
    parser.add_argument(
        "--vqa_test_questions_json_path",
        type=str,
        help="Path to the json file with all the vqa test questions.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to store the filled json.",
    )
    args = parser.parse_args()

    if args.dataset == "vqav2":
        fill_vqav2_test_json(
            args.input_path,
            args.output_path,
            args.vqa_test_questions_json_path,
        )
    else:
        fill_vizwiz_test_json(
            args.input_path,
            args.output_path,
            args.vqa_test_questions_json_path,
        )
