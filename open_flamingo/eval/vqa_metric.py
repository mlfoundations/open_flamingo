import copy
import datetime
import json
import os
import random
import re
import sys

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"


class VQA:
    def __init__(self, annotation_file=None, question_file=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if not annotation_file == None and not question_file == None:
            print("loading VQA annotations and questions into memory...")
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, "r"))
            questions = json.load(open(question_file, "r"))
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        imgToQA = {ann["image_id"]: [] for ann in self.dataset["annotations"]}
        qa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        qqa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        for ann in self.dataset["annotations"]:
            imgToQA[ann["image_id"]] += [ann]
            qa[ann["question_id"]] = ann
        for ques in self.questions["questions"]:
            qqa[ques["question_id"]] = ques
        print("index created!")

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.dataset["info"].items():
            print("%s: %s" % (key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                        quesTypes (str array)   : get question ids for given question types
                        ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                anns = sum(
                    [self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA],
                    [],
                )
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(quesTypes) == 0
                else [ann for ann in anns if ann["question_type"] in quesTypes]
            )
            anns = (
                anns
                if len(ansTypes) == 0
                else [ann for ann in anns if ann["answer_type"] in ansTypes]
            )
        ids = [ann["question_id"] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
         Get image ids that satisfy given filter conditions. default skips that filter
         :param quesIds   (int array)   : get image ids for given question ids
        quesTypes (str array)   : get image ids for given question types
        ansTypes  (str array)   : get image ids for given answer types
         :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(quesIds) == 0:
                anns = sum(
                    [self.qa[quesId] for quesId in quesIds if quesId in self.qa], []
                )
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(quesTypes) == 0
                else [ann for ann in anns if ann["question_type"] in quesTypes]
            )
            anns = (
                anns
                if len(ansTypes) == 0
                else [ann for ann in anns if ann["answer_type"] in ansTypes]
            )
        ids = [ann["image_id"] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann["question_id"]
            print("Question: %s" % (self.qqa[quesId]["question"]))
            for ans in ann["answers"]:
                print("Answer %d: %s" % (ans["answer_id"], ans["answer"]))

    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = VQA()
        res.questions = json.load(open(quesFile))
        res.dataset["info"] = copy.deepcopy(self.questions["info"])
        res.dataset["task_type"] = copy.deepcopy(self.questions["task_type"])
        res.dataset["data_type"] = copy.deepcopy(self.questions["data_type"])
        res.dataset["data_subtype"] = copy.deepcopy(self.questions["data_subtype"])
        res.dataset["license"] = copy.deepcopy(self.questions["license"])

        print("Loading and preparing results...     ")
        time_t = datetime.datetime.utcnow()
        anns = json.load(open(resFile))
        assert type(anns) == list, "results is not an array of objects"
        annsQuesIds = [ann["question_id"] for ann in anns]
        # print set of question ids that do not have corresponding annotations

        # assert set(annsQuesIds) == set(self.getQuesIds()), \
        # 'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file.'
        for ann in anns:
            quesId = ann["question_id"]
            if res.dataset["task_type"] == "Multiple Choice":
                assert (
                    ann["answer"] in self.qqa[quesId]["multiple_choices"]
                ), "predicted answer is not one of the multiple choices"
            qaAnn = self.qa[quesId]
            ann["image_id"] = qaAnn["image_id"]
            ann["question_type"] = qaAnn["question_type"]
            if "answer_type" in ann:
                ann["answer_type"] = qaAnn["answer_type"]
        print(
            "DONE (t=%0.2fs)" % ((datetime.datetime.utcnow() - time_t).total_seconds())
        )

        res.dataset["annotations"] = anns
        res.createIndex()
        return res


class VQAEval:
    def __init__(self, vqa, vqaRes, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        if not vqa is None and not vqaRes is None:
            self.params = {"question_id": vqaRes.getQuesIds()}
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params["question_id"]]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        accQuesType = {}
        accAnsType = {}
        print("computing accuracy")
        step = 0
        for quesId in quesIds:
            for ansDic in gts[quesId]["answers"]:
                ansDic["answer"] = ansDic["answer"].replace("\n", " ")
                ansDic["answer"] = ansDic["answer"].replace("\t", " ")
                ansDic["answer"] = ansDic["answer"].strip()
            resAns = res[quesId]["answer"]
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []

            for ansDic in gts[quesId]["answers"]:
                ansDic["answer"] = self.processPunctuation(ansDic["answer"])
                ansDic["answer"] = self.processDigitArticle(ansDic["answer"])

            for gtAnsDatum in gts[quesId]["answers"]:
                otherGTAns = [
                    item for item in gts[quesId]["answers"] if item != gtAnsDatum
                ]
                matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            quesType = gts[quesId]["question_type"]
            ansType = (
                gts[quesId]["answer_type"] if "answer_type" in gts[quesId] else "other"
            )
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            if step % 100 == 0:
                self.updateProgress(step / float(len(quesIds)))
            step = step + 1

        self.setAccuracy(accQA, accQuesType, accAnsType)
        print("Done computing accuracy")

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy["overall"] = round(100 * float(sum(accQA)) / len(accQA), self.n)
        self.accuracy["perQuestionType"] = {
            quesType: round(
                100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]),
                self.n,
            )
            for quesType in accQuesType
        }
        self.accuracy["perAnswerType"] = {
            ansType: round(
                100 * float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n
            )
            for ansType in accAnsType
        }

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rFinshed Percent: [{0}] {1}% {2}".format(
            "#" * block + "-" * (barLength - block), int(progress * 100), status
        )
        sys.stdout.write(text)
        sys.stdout.flush()


def compute_vqa_accuracy(result_json_path, question_json_path, annotation_json_path):
    """Compute the VQA accuracy metric.

    Args:
        result_json_path (str): Path to the json file with model outputs
        question_json_path (str): Path to the json file with questions
        annotation_json_path (str): Path to the json file with annotations

    Returns:
        float: VQA accuracy
    """

    # create vqa object and vqaRes object
    vqa = VQA(annotation_json_path, question_json_path)
    vqaRes = vqa.loadRes(result_json_path, question_json_path)

    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval = VQAEval(vqa, vqaRes, n=2)

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()

    return vqaEval.accuracy["overall"]


def postprocess_vqa_generation(predictions):
    answer = re.split("Question|Answer|Short", predictions, 1)[0]
    answer = re.split(", ", answer, 1)[0]
    return answer
