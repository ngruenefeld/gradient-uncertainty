import json
import openai

REPHRASE_PROMPT_TEMPLATES = {
    "en": {
        "system": (
            "You are an expert in paraphrasing text. Your task is to generate {num} distinct rephrasings "
            "of the provided text while maintaining the original meaning. Ensure that each rephrasing is "
            "unique and not a simple repetition of the original."
        ),
        "user": {
            "low": (
                "Generate {num} paraphrases of the following text:\n\n{text}\n\n"
                "Make only minor lexical substitutions. Keep sentence structure and phrasing nearly identical."
            ),
            "medium": (
                "Generate {num} paraphrases of the following text:\n\n{text}\n\n"
                "Change vocabulary and some sentence structures, while keeping the same overall message."
            ),
            "high": (
                "Generate {num} paraphrases of the following text:\n\n{text}\n\n"
                "Drastically rephrase and restructure the content. Be creative in how the message is conveyed, but ensure the core meaning is preserved."
            ),
        },
    },
    "de": {
        "system": (
            "Du bist ein Experte für das Paraphrasieren von Texten. Deine Aufgabe ist es, {num} verschiedene Umschreibungen "
            "des bereitgestellten Textes zu erstellen, wobei die ursprüngliche Bedeutung beibehalten werden soll. Jede Umschreibung "
            "muss einzigartig sein und darf den Originaltext nicht einfach wiederholen."
        ),
        "user": {
            "low": (
                "Erzeuge {num} Umschreibungen des folgenden Textes:\n\n{text}\n\n"
                "Nur minimale Wortänderungen. Satzstruktur und Formulierung bleiben fast unverändert."
            ),
            "medium": (
                "Erzeuge {num} Umschreibungen des folgenden Textes:\n\n{text}\n\n"
                "Verändere Wortwahl und teilweise die Satzstruktur, aber behalte die Hauptaussage bei."
            ),
            "high": (
                "Erzeuge {num} Umschreibungen des folgenden Textes:\n\n{text}\n\n"
                "Formuliere stark um und strukturiere den Inhalt kreativ um. Die Kernbedeutung muss erhalten bleiben."
            ),
        },
    },
    "es": {
        "system": (
            "Eres un experto en parafrasear textos. Tu tarea es generar {num} reformulaciones distintas del texto proporcionado, "
            "manteniendo el significado original. Cada reformulación debe ser única y no simplemente repetir el texto original."
        ),
        "user": {
            "low": (
                "Genera {num} paráfrasis del siguiente texto:\n\n{text}\n\n"
                "Realiza solo pequeños cambios léxicos. Mantén la estructura de las oraciones casi igual."
            ),
            "medium": (
                "Genera {num} paráfrasis del siguiente texto:\n\n{text}\n\n"
                "Cambia vocabulario y algunas estructuras de oración, pero conserva el mensaje general."
            ),
            "high": (
                "Genera {num} paráfrasis del siguiente texto:\n\n{text}\n\n"
                "Reformula y reestructura completamente el contenido. Sé creativo sin alterar el significado esencial."
            ),
        },
    },
    "fr": {
        "system": (
            "Vous êtes un expert en reformulation de texte. Votre tâche est de générer {num} reformulations distinctes "
            "du texte fourni tout en maintenant le sens original. Chaque reformulation doit être unique et ne pas simplement "
            "répéter le texte original."
        ),
        "user": {
            "low": (
                "Générez {num} reformulations du texte suivant :\n\n{text}\n\n"
                "Effectuez uniquement des substitutions lexicales mineures. Gardez une structure similaire."
            ),
            "medium": (
                "Générez {num} reformulations du texte suivant :\n\n{text}\n\n"
                "Changez le vocabulaire et certaines structures tout en conservant le message principal."
            ),
            "high": (
                "Générez {num} reformulations du texte suivant :\n\n{text}\n\n"
                "Reformulez et réorganisez le texte en profondeur, en préservant le sens global."
            ),
        },
    },
    "it": {
        "system": (
            "Sei un esperto nella parafrasi dei testi. Il tuo compito è generare {num} riformulazioni distinte del testo fornito "
            "mantenendo il significato originale. Ogni riformulazione deve essere unica e non una semplice ripetizione."
        ),
        "user": {
            "low": (
                "Genera {num} parafrasi del seguente testo:\n\n{text}\n\n"
                "Modifica solo poche parole. Mantieni struttura e sintassi quasi invariate."
            ),
            "medium": (
                "Genera {num} parafrasi del seguente testo:\n\n{text}\n\n"
                "Cambia parole e alcune strutture, mantenendo il significato centrale."
            ),
            "high": (
                "Genera {num} parafrasi del seguente testo:\n\n{text}\n\n"
                "Riformula in modo significativo. Ristruttura il contenuto con creatività senza alterare il significato."
            ),
        },
    },
    "ko": {
        "system": (
            "당신은 텍스트를 바꾸는 전문가입니다. 당신의 임무는 제공된 텍스트의 원래 의미를 유지하면서 {num}가지 독특한 바꾸기를 생성하는 것입니다. 각 바꾸기가 독특하고 원본 텍스트를 단순히 반복하지 않도록 하십시오."
        ),
        "user": {
            "low": (
                "다음 텍스트의 {num}가지 바꾸기를 생성하십시오:\n\n{text}\n\n"
                "단어 수준에서 약간의 변경만 하십시오. 문장 구조는 거의 그대로 유지하십시오."
            ),
            "medium": (
                "다음 텍스트의 {num}가지 바꾸기를 생성하십시오:\n\n{text}\n\n"
                "다양한 단어와 문장 구조를 사용하되 의미는 유지하십시오."
            ),
            "high": (
                "다음 텍스트의 {num}가지 바꾸기를 생성하십시오:\n\n{text}\n\n"
                "창의적으로 재구성하고 표현을 완전히 바꾸되, 핵심 의미는 그대로 유지하십시오."
            ),
        },
    },
    "pt": {
        "system": (
            "Você é um especialista em parafrasear textos. Sua tarefa é gerar {num} reformulações distintas do texto fornecido, mantendo o significado original. Cada reformulação deve ser única e não apenas repetir o texto original."
        ),
        "user": {
            "low": (
                "Gere {num} paráfrases do seguinte texto:\n\n{text}\n\n"
                "Faça apenas substituições leves de palavras. Mantenha a estrutura original."
            ),
            "medium": (
                "Gere {num} paráfrases do seguinte texto:\n\n{text}\n\n"
                "Altere o vocabulário e parte da estrutura, preservando o sentido."
            ),
            "high": (
                "Gere {num} paráfrases do seguinte texto:\n\n{text}\n\n"
                "Reestruture e reformule de forma criativa. Preserve o significado essencial."
            ),
        },
    },
    "ru": {
        "system": (
            "Вы являетесь экспертом в перефразировании текста. Ваша задача — создать {num} различных перефразировок предоставленного текста, сохраняя оригинальное значение. Каждая перефразировка должна быть уникальной и не дублировать исходный текст."
        ),
        "user": {
            "low": (
                "Создайте {num} перефразировок следующего текста:\n\n{text}\n\n"
                "Измените только отдельные слова. Структура предложений должна остаться почти такой же."
            ),
            "medium": (
                "Создайте {num} перефразировок следующего текста:\n\n{text}\n\n"
                "Измените лексику и частично структуру, сохранив общий смысл."
            ),
            "high": (
                "Создайте {num} перефразировок следующего текста:\n\n{text}\n\n"
                "Кардинально переформулируйте и перестройте текст, не изменяя его суть."
            ),
        },
    },
    "zh": {
        "system": (
            "你是一个文本改写的专家。你的任务是生成提供的文本的{num}种不同的改写，同时保持原意。确保每个改写都是独特的，而不是简单地重复原始文本。"
        ),
        "user": {
            "low": (
                "生成以下文本的{num}种改写：\n\n{text}\n\n"
                "仅做轻微的词语替换，语序和句式基本保持一致。"
            ),
            "medium": (
                "生成以下文本的{num}种改写：\n\n{text}\n\n"
                "更换部分词语和语句结构，保留原始含义。"
            ),
            "high": (
                "生成以下文本的{num}种改写：\n\n{text}\n\n"
                "可大幅度重构和重新表达内容，只要保留主要意思即可。"
            ),
        },
    },
}


def rephrase_text(
    text_to_rephrase,
    client,
    model,
    language="en",
    number_of_rephrasings=3,
    divergence="medium",
):
    supported_languages = REPHRASE_PROMPT_TEMPLATES.keys()

    if language not in supported_languages:
        raise ValueError(
            f"Language '{language}' is not supported. Supported languages are: {list(supported_languages)}"
        )

    if not isinstance(number_of_rephrasings, int) or number_of_rephrasings < 1:
        raise ValueError("number_of_rephrasings must be a positive integer")

    if divergence not in ["low", "medium", "high"]:
        raise ValueError("divergence must be 'low', 'medium', or 'high'")

    system_prompt = REPHRASE_PROMPT_TEMPLATES[language]["system"].format(
        num=number_of_rephrasings
    )
    user_prompt = REPHRASE_PROMPT_TEMPLATES[language]["user"][divergence].format(
        text=text_to_rephrase, num=number_of_rephrasings
    )

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
            return json.loads(response.output_text)
        except json.JSONDecodeError as json_error:
            print(f"JSONDecodeError: {json_error}")
            return {"error": "Invalid JSON response from API"}
    except openai.BadRequestError as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}


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
