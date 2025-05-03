import json
import openai


def rephrase_text(text_to_rephrase, client, model, language="en"):
    supported_languages = [
        "en",
        "de",
        "es",
        "fr",
        "it",
        "ko",
        "pt",
        "ru",
        "zh",
    ]

    if language not in supported_languages:
        raise ValueError(
            f"Language '{language}' is not supported. Supported languages are: {supported_languages}"
        )
    elif language == "en":
        system_prompt = "You are an expert in paraphrasing text. Your task is to generate three distinct rephrasings of the provided text while maintaining the original meaning. Ensure that each rephrasing is unique and does not simply repeat the original text."
        user_prompt = (
            f"Generate three paraphrases of the following text:\n\n{text_to_rephrase}\n\n"
            "Do not divert from the original meaning in any way. Use different words and sentence structures to convey the same message.\n"
            "Make sure to provide three distinct rephrasings."
        )
    elif language == "de":
        system_prompt = "Du bist ein Experte für das Paraphrasieren von Texten. Deine Aufgabe ist es, drei verschiedene Umschreibungen des bereitgestellten Textes zu erstellen, während die ursprüngliche Bedeutung beibehalten wird. Stelle sicher, dass jede Umschreibung einzigartig ist und den Originaltext nicht einfach wiederholt."
        user_prompt = (
            f"Erzeuge drei Umschreibungen des folgenden Textes:\n\n{text_to_rephrase}\n\n"
            "Weiche in keiner Weise von der ursprünglichen Bedeutung ab. Verwende andere Wörter und Satzstrukturen, um dieselbe Botschaft zu vermitteln.\n"
            "Stelle sicher, dass du drei verschiedene Umschreibungen angibst."
        )
    elif language == "es":
        system_prompt = "Eres un experto en parafrasear textos. Tu tarea es generar tres reformulaciones distintas del texto proporcionado, manteniendo el significado original. Asegúrate de que cada reformulación sea única y no repita simplemente el texto original."
        user_prompt = (
            f"Genera tres paráfrasis del siguiente texto:\n\n{text_to_rephrase}\n\n"
            "No te desvíes del significado original de ninguna manera. Usa diferentes palabras y estructuras de oraciones para transmitir el mismo mensaje.\n"
            "Asegúrate de proporcionar tres reformulaciones distintas."
        )
    elif language == "fr":
        system_prompt = "Vous êtes un expert en reformulation de texte. Votre tâche consiste à générer trois reformulations distinctes du texte fourni tout en maintenant le sens original. Assurez-vous que chaque reformulation est unique et ne répète pas simplement le texte original."
        user_prompt = (
            f"Générez trois reformulations du texte suivant:\n\n{text_to_rephrase}\n\n"
            "Ne vous écartez en aucun cas du sens original. Utilisez des mots et des structures de phrases différents pour transmettre le même message.\n"
            "Assurez-vous de fournir trois reformulations distinctes."
        )
    elif language == "it":
        system_prompt = "Sei un esperto nella parafrasi di testi. Il tuo compito è generare tre riformulazioni distinte del testo fornito mantenendo il significato originale. Assicurati che ogni riformulazione sia unica e non ripeta semplicemente il testo originale."
        user_prompt = (
            f"Genera tre parafrasi del seguente testo:\n\n{text_to_rephrase}\n\n"
            "Non deviare in alcun modo dal significato originale. Usa parole e strutture di frase diverse per trasmettere lo stesso messaggio.\n"
            "Assicurati di fornire tre riformulazioni distinte."
        )
    elif language == "ko":
        system_prompt = "당신은 텍스트를 바꾸는 전문가입니다. 당신의 임무는 제공된 텍스트의 원래 의미를 유지하면서 세 가지 독특한 바꾸기를 생성하는 것입니다. 각 바꾸기가 독특하고 원본 텍스트를 단순히 반복하지 않도록 하십시오."
        user_prompt = (
            f"다음 텍스트의 세 가지 바꾸기를 생성하십시오:\n\n{text_to_rephrase}\n\n"
            "어떤 식으로든 원래 의미에서 벗어나지 마십시오. 동일한 메시지를 전달하기 위해 다른 단어와 문장 구조를 사용하십시오.\n"
            "세 가지 독특한 바꾸기를 제공해야 합니다."
        )
    elif language == "pt":
        system_prompt = "Você é um especialista em parafrasear textos. Sua tarefa é gerar três reformulações distintas do texto fornecido, mantendo o significado original. Certifique-se de que cada reformulação seja única e não repita simplesmente o texto original."
        user_prompt = (
            f"Gere três paráfrases do seguinte texto:\n\n{text_to_rephrase}\n\n"
            "Não se desvie do significado original de forma alguma. Use palavras e estruturas de frases diferentes para transmitir a mesma mensagem.\n"
            "Certifique-se de fornecer três reformulações distintas."
        )
    elif language == "ru":
        system_prompt = "Вы являетесь экспертом в перефразировании текста. Ваша задача - создать три различных перефразировки предоставленного текста, сохраняя оригинальное значение. Убедитесь, что каждая перефразировка уникальна и не просто повторяет оригинальный текст."
        user_prompt = (
            f"Создайте три перефразировки следующего текста:\n\n{text_to_rephrase}\n\n"
            "Ни в коем случае не отклоняйтесь от оригинального смысла. Используйте разные слова и структуры предложений, чтобы передать то же сообщение.\n"
            "Убедитесь, что вы предоставили три различных перефразировки."
        )
    elif language == "zh":
        system_prompt = "你是一个文本改写的专家。你的任务是生成提供的文本的三种不同的改写，同时保持原意。确保每个改写都是独特的，而不是简单地重复原始文本。"
        user_prompt = (
            f"生成以下文本的三种改写：\n\n{text_to_rephrase}\n\n"
            "无论如何都不要偏离原意。使用不同的单词和句子结构来传达相同的信息。\n"
            "确保提供三种不同的改写。"
        )
    else:
        raise ValueError(
            f"Language '{language}' is not supported. Supported languages are: {supported_languages}"
        )

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
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
