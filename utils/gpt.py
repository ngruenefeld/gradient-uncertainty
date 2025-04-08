import json
import openai


def rephrase_text(text_to_rephrase, client, model):
    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You are an expert in paraphrasing text. Your task is to generate three distinct rephrasings of the provided text while maintaining the original meaning. Ensure that each rephrasing is unique and does not simply repeat the original text.",
                },
                {
                    "role": "user",
                    "content": f"Generate three paraphrases of the following text:\n\n{text_to_rephrase}\n\nDo not divert from the original meaning in any way. Use different words and sentence structures to convey the same message.\nMake sure to provide three distinct rephrasings.",
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
        try:
            # Attempt to parse the response as JSON
            event = json.loads(response.output_text)
            return event
        except json.JSONDecodeError as json_error:
            # Handle JSON parsing errors
            print(f"JSONDecodeError: {json_error}")
            return {"error": "Invalid JSON response from API"}
    except openai.BadRequestError as e:
        error_message = str(e)  # Extract the error message as a string
        print(f"Error: {error_message}")
        return {"error": error_message}


def evaluate_answers(question, answer, reference_answers, client, model):
    reference_answers_formatted = "\n".join(reference_answers)
    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You are an assistant that evaluates whether answer candidates contain the correct information based on reference answers.",
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
        try:
            # Attempt to parse the response as JSON
            event = json.loads(response.output_text)
            return event
        except json.JSONDecodeError as json_error:
            # Handle JSON parsing errors
            print(f"JSONDecodeError: {json_error}")
            return {"error": "Invalid JSON response from API"}
    except openai.BadRequestError as e:
        error_message = str(e)  # Extract the error message as a string
        print(f"Error: {error_message}")
        return {"error": error_message}
