import json


def rephrase_text(text_to_rephrase, client, model):
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a helpful assistant that paraphrases text.",
            },
            {
                "role": "user",
                "content": f"Generate three paraphrases of the following text: {text_to_rephrase}",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "rephrasings_list",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "rephrasings": {
                            "type": "array",
                            "description": "A list of rephrased versions of the original input.",
                            "items": {
                                "type": "string",
                                "description": "A single rephrased sentence.",
                            },
                        },
                    },
                    "required": ["rephrasings"],
                    "additionalProperties": False,
                },
            }
        },
    )
    event = json.loads(response.output_text)
    return event


def evaluate_answers(question, answer, reference_answers, client, model):
    reference_answers_formatted = "\n".join(reference_answers)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a helpful assistant that evaluates question-answer pairs.",
            },
            {
                "role": "user",
                "content": f"Your task is to evaluate whether a generated answer candidate to a given question is correct or not, given a set of correct reference answers.\nThe question is: {question}\nThe generated answer candidate is: {answer}\nThe correct reference answers are:\n{reference_answers_formatted}\nIs the answer candidate correct or not? The answer might be overly verbose, try to extract what is meant.",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "answer_verification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_correct": {
                            "type": "boolean",
                            "description": "Indicates whether the given answer matches any of the correct answers.",
                        },
                    },
                    "required": [
                        "is_correct",
                    ],
                    "additionalProperties": False,
                },
            }
        },
    )
    event = json.loads(response.output_text)
    return event
