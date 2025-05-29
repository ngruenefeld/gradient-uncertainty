import json
import openai


def rephrase_text(
    text_to_rephrase, client, model, language="en", number_of_rephrasings=3
):
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

    if not isinstance(number_of_rephrasings, int) or number_of_rephrasings < 1:
        raise ValueError("number_of_rephrasings must be a positive integer")

    prompts = {
        "en": {
            "system": f"You are an expert in paraphrasing text. Your task is to generate {number_of_rephrasings} distinct rephrasings of the provided text while maintaining the original meaning. Ensure that each rephrasing is unique and does not simply repeat the original text.",
            "user": f"Generate {number_of_rephrasings} paraphrases of the following text:\n\n{text_to_rephrase}\n\n"
            f"Do not divert from the original meaning in any way. Use different words and sentence structures to convey the same message.\n"
            f"Make sure to provide {number_of_rephrasings} distinct rephrasings.",
        },
        "de": {
            "system": f"Du bist ein Experte für das Paraphrasieren von Texten. Deine Aufgabe ist es, {number_of_rephrasings} verschiedene Umschreibungen des bereitgestellten Textes zu erstellen, während die ursprüngliche Bedeutung beibehalten wird. Stelle sicher, dass jede Umschreibung einzigartig ist und den Originaltext nicht einfach wiederholt.",
            "user": f"Erzeuge {number_of_rephrasings} Umschreibungen des folgenden Textes:\n\n{text_to_rephrase}\n\n"
            f"Weiche in keiner Weise von der ursprünglichen Bedeutung ab. Verwende andere Wörter und Satzstrukturen, um dieselbe Botschaft zu vermitteln.\n"
            f"Stelle sicher, dass du {number_of_rephrasings} verschiedene Umschreibungen angibst.",
        },
        "es": {
            "system": f"Eres un experto en parafrasear textos. Tu tarea es generar {number_of_rephrasings} reformulaciones distintas del texto proporcionado, manteniendo el significado original. Asegúrate de que cada reformulación sea única y no repita simplemente el texto original.",
            "user": f"Genera {number_of_rephrasings} paráfrasis del siguiente texto:\n\n{text_to_rephrase}\n\n"
            f"No te desvíes del significado original de ninguna manera. Usa diferentes palabras y estructuras de oraciones para transmitir el mismo mensaje.\n"
            f"Asegúrate de proporcionar {number_of_rephrasings} reformulaciones distintas.",
        },
        "fr": {
            "system": f"Vous êtes un expert en reformulation de texte. Votre tâche consiste à générer {number_of_rephrasings} reformulations distinctes du texte fourni tout en maintenant le sens original. Assurez-vous que chaque reformulation est unique et ne répète pas simplement le texte original.",
            "user": f"Générez {number_of_rephrasings} reformulations du texte suivant:\n\n{text_to_rephrase}\n\n"
            f"Ne vous écartez en aucun cas du sens original. Utilisez des mots et des structures de phrases différents pour transmettre le même message.\n"
            f"Assurez-vous de fournir {number_of_rephrasings} reformulations distinctes.",
        },
        "it": {
            "system": f"Sei un esperto nella parafrasi di testi. Il tuo compito è generare {number_of_rephrasings} riformulazioni distinte del testo fornito mantenendo il significato originale. Assicurati che ogni riformulazione sia unica e non ripeta semplicemente il testo originale.",
            "user": f"Genera {number_of_rephrasings} parafrasi del seguente testo:\n\n{text_to_rephrase}\n\n"
            f"Non deviare in alcun modo dal significato originale. Usa parole e strutture di frase diverse per trasmettere lo stesso messaggio.\n"
            f"Assicurati di fornire {number_of_rephrasings} riformulazioni distinte.",
        },
        "ko": {
            "system": f"당신은 텍스트를 바꾸는 전문가입니다. 당신의 임무는 제공된 텍스트의 원래 의미를 유지하면서 {number_of_rephrasings}가지 독특한 바꾸기를 생성하는 것입니다. 각 바꾸기가 독특하고 원본 텍스트를 단순히 반복하지 않도록 하십시오.",
            "user": f"다음 텍스트의 {number_of_rephrasings}가지 바꾸기를 생성하십시오:\n\n{text_to_rephrase}\n\n"
            f"어떤 식으로든 원래 의미에서 벗어나지 마십시오. 동일한 메시지를 전달하기 위해 다른 단어와 문장 구조를 사용하십시오.\n"
            f"{number_of_rephrasings}가지 독특한 바꾸기를 제공해야 합니다.",
        },
        "pt": {
            "system": f"Você é um especialista em parafrasear textos. Sua tarefa é gerar {number_of_rephrasings} reformulações distintas do texto fornecido, mantendo o significado original. Certifique-se de que cada reformulação seja única e não repita simplesmente o texto original.",
            "user": f"Gere {number_of_rephrasings} paráfrases do seguinte texto:\n\n{text_to_rephrase}\n\n"
            f"Não se desvie do significado original de forma alguma. Use palavras e estruturas de frases diferentes para transmitir a mesma mensagem.\n"
            f"Certifique-se de fornecer {number_of_rephrasings} reformulações distintas.",
        },
        "ru": {
            "system": f"Вы являетесь экспертом в перефразировании текста. Ваша задача - создать {number_of_rephrasings} различных перефразировки предоставленного текста, сохраняя оригинальное значение. Убедитесь, что каждая перефразировка уникальна и не просто повторяет оригинальный текст.",
            "user": f"Создайте {number_of_rephrasings} перефразировки следующего текста:\n\n{text_to_rephrase}\n\n"
            f"Ни в коем случае не отклоняйтесь от оригинального смысла. Используйте разные слова и структуры предложений, чтобы передать то же сообщение.\n"
            f"Убедитесь, что вы предоставили {number_of_rephrasings} различных перефразировки.",
        },
        "zh": {
            "system": f"你是一个文本改写的专家。你的任务是生成提供的文本的{number_of_rephrasings}种不同的改写，同时保持原意。确保每个改写都是独特的，而不是简单地重复原始文本。",
            "user": f"生成以下文本的{number_of_rephrasings}种改写：\n\n{text_to_rephrase}\n\n"
            f"无论如何都不要偏离原意。使用不同的单词和句子结构来传达相同的信息。\n"
            f"确保提供{number_of_rephrasings}种不同的改写。",
        },
    }

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": prompts[language]["system"],
                },
                {
                    "role": "user",
                    "content": prompts[language]["user"],
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
                                "description": f"A list of {number_of_rephrasings} rephrased versions of the original input.",
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
                    "content": (
                        "You are a fact-checking evaluator. You decide whether an answer contains the same specific factual content "
                        "as one of a set of reference answers. You focus only on the core factual entity mentioned — such as a number, name, or place."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
        You will be given a question, a generated answer, and a list of reference answers.

        Determine whether the answer correctly contains the **same factual answer** (e.g., a number, name, location, etc.) as one of the references.

        ### Accept the answer if:
        - The correct fact is **clearly present**, even if the answer adds irrelevant or incorrect background.
        - The answer uses different words or phrasing to convey the same meaning.

        ### Reject the answer if:
        - It states a different fact or contradicts the correct one.
        - It introduces confusion or falsehood **that changes the meaning of the core fact**.

        ---

        **Question**: {question}  
        **Generated Answer**: {answer}  
        **Reference Answers**: {reference_answers_formatted}

        Respond only in JSON:
        {{"is_correct": true}} or {{"is_correct": false}}
        """,
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
                                "description": "True if the core factual answer (e.g., number or name) is included and not contradicted.",
                            },
                        },
                        "required": ["is_correct"],
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
