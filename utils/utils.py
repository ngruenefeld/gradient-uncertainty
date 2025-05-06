from datasets import load_dataset, DatasetDict, Dataset
import torch
import gc
import nltk
import random
import requests
from nltk.corpus import stopwords, wordnet
from deep_translator import GoogleTranslator
from functools import lru_cache


nltk.download("stopwords")
nltk.download("wordnet")


ISO_639_1_TO_3 = {
    "en": "eng",
    "de": None,
    "es": "spa",
    "fr": "fra",
    "it": "ita",
    "ko": None,
    "pt": "por",
    "ru": None,
    "zh": "cmn",
}


def get_response(prompt, model, tokenizer, device):
    try:
        model.eval()

        # Use no_grad to prevent gradient storage during inference
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=100)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated_text.startswith(prompt):
            completion = generated_text[len(prompt) :].strip()
        else:
            completion = generated_text

        # Free memory
        del inputs, outputs
        torch.cuda.empty_cache()

        return completion
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        torch.cuda.empty_cache()
        return {"error": str(e)}


def arithmetic_mean_change(param_values, param_grads):
    new_param_values = param_values + param_grads
    denominator = 0.5 * (param_values + new_param_values)

    # Small epsilon to avoid division by zero
    epsilon = 1e-8

    # Calculate symmetric percent change
    normalized_grads = torch.where(
        denominator.abs() > epsilon,
        param_grads / denominator,
        torch.zeros_like(param_grads),
    )

    return normalized_grads


def completion_gradient(
    prompt,
    completion,
    model,
    tokenizer,
    device,
    response_only=True,
    normalize=False,
    max_length=None,
):
    try:
        model.train()

        full_text = prompt + completion

        # Get the encodings for both prompt and full text with truncation
        if max_length:
            full_encodings = tokenizer(
                full_text, return_tensors="pt", max_length=max_length, truncation=True
            )
        else:
            full_encodings = tokenizer(full_text, return_tensors="pt")
        input_ids = full_encodings.input_ids.to(device)

        if max_length:
            prompt_encodings = tokenizer(
                prompt, return_tensors="pt", max_length=max_length, truncation=True
            )
        else:
            prompt_encodings = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_encodings.input_ids.shape[1]

        # Normal processing
        labels = input_ids.clone()
        if response_only:
            labels[0, :prompt_len] = -100  # Ignore loss for prompt tokens

        # Calculate loss and backprop
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        model.zero_grad()
        loss.backward()

        # Calculate gradient norm
        num_params = sum(p.numel() for _, p in model.named_parameters())
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if normalize == False:
                    param_norm = param.grad.detach().norm(2)
                else:
                    param_values = param.detach()
                    param_grads = param.grad.detach()

                    normalized_grads = arithmetic_mean_change(param_values, param_grads)
                    param_norm = normalized_grads.detach().norm(2)

                total_norm += param_norm.item() ** 2

        uncertainty = torch.tensor(total_norm**0.5)

        # Calculate completion length
        completion_length = (
            full_encodings.input_ids.shape[1] - prompt_encodings.input_ids.shape[1]
        )

        # Free memory before returning
        del outputs, loss, input_ids, labels, full_encodings, prompt_encodings
        gc.collect()
        torch.cuda.empty_cache()

        return uncertainty, completion_length
    except Exception as e:
        print(f"Error in completion_gradient: {str(e)}")
        # Make sure to free memory
        torch.cuda.empty_cache()
        gc.collect()
        return {"error": str(e)}


def bert_gradient(inputs, labels, model, normalize=False):
    try:
        model.train()

        model.zero_grad()

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()

        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if normalize == False:
                    param_norm = param.grad.detach().norm(2)
                else:
                    param_values = param.detach()
                    param_grads = param.grad.detach()

                    normalized_grads = arithmetic_mean_change(param_values, param_grads)
                    param_norm = normalized_grads.detach().norm(2)

                total_norm += param_norm.item() ** 2

        uncertainty = torch.tensor(total_norm**0.5)

        del outputs, loss, inputs, labels
        gc.collect()
        torch.cuda.empty_cache()

        return uncertainty
    except Exception as e:
        print(f"Error in bert_gradient: {str(e)}")
        # Make sure to free memory
        torch.cuda.empty_cache()
        gc.collect()
        return {"error": str(e)}


def load_bert_dataset_dicts(choice="ag_news"):
    if choice == "ag_news":
        dataset = load_dataset("fancyzhx/ag_news")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        label_names = train_dataset.features["label"].names

        train_labels = [label_names[label] for label in train_dataset["label"]]
        test_labels = [label_names[label] for label in test_dataset["label"]]

        train_data = {
            "text": train_dataset["text"],
            "origin": ["ag_news"] * len(train_dataset["text"]),
            "label": train_labels,
        }

        test_data = {
            "text": test_dataset["text"],
            "origin": ["ag_news"] * len(test_dataset["text"]),
            "label": test_labels,
        }

        return train_data, test_data

    elif choice == "pubmed":
        dataset = load_dataset("MedRAG/pubmed", streaming=True)["train"]

        sample_size = 10000

        content_samples = []
        count = 0

        for example in dataset:
            if count >= sample_size:
                break
            content_samples.append(example["content"])
            count += 1

        data = {
            "text": content_samples,
            "origin": ["pubmed"] * len(content_samples),
            "label": ["Medicine"] * len(content_samples),
        }

        return data

    elif choice == "mmlu":
        cs_dataset = load_dataset("tasksource/mmlu", "computer_security")
        phil_dataset = load_dataset("tasksource/mmlu", "philosophy")

        cs_dataset_val = cs_dataset["validation"]
        cs_dataset_test = cs_dataset["validation"]
        phil_dataset = phil_dataset["test"]

        cs_data_val = {
            "text": cs_dataset_val["question"],
            "origin": ["mmlu"] * len(cs_dataset_val["question"]),
            "label": ["Computer Security"] * len(cs_dataset_val["question"]),
        }

        cs_data_test = {
            "text": cs_dataset_test["question"],
            "origin": ["mmlu"] * len(cs_dataset_test["question"]),
            "label": ["Computer Security"] * len(cs_dataset_test["question"]),
        }

        phil_data = {
            "text": phil_dataset["question"],
            "origin": ["mmlu"] * len(phil_dataset["question"]),
            "label": ["Philosophy"] * len(phil_dataset["question"]),
        }

        return cs_data_val, cs_data_test, phil_data

    elif choice == "scienceqa":
        dataset = load_dataset("derek-thomas/ScienceQA")

        dataset_train = dataset["train"].filter(
            lambda example: example["lecture"] and example["lecture"] != ""
        )
        dataset_test = dataset["test"].filter(
            lambda example: example["lecture"] and example["lecture"] != ""
        )

        data_train = {
            "text": dataset_train["lecture"],
            "origin": ["ScienceQA"] * len(dataset_train["lecture"]),
            "label": dataset_train["topic"],
        }

        data_test = {
            "text": dataset_test["lecture"],
            "origin": ["ScienceQA"] * len(dataset_test["lecture"]),
            "label": dataset_test["topic"],
        }

        return data_train, data_test

    elif choice == "legalqa":
        dataset = load_dataset("isaacus/open-australian-legal-qa")["train"]

        data = {
            "text": dataset["text"],
            "origin": ["ScienceQA"] * len(dataset["text"]),
            "label": ["Legal"] * len(dataset["text"]),
        }

        return data

    else:
        raise ValueError(f"Dataset {choice} not supported.")


def load_bert_datasets(choice="ag_news"):
    if choice == "ag_news":
        train_data, test_data = load_bert_dataset_dicts("ag_news")

        indices_to_remove = {
            i for i, v in enumerate(train_data["label"]) if v != "Sports"
        }

        filtered_train_data = {
            key: [v for i, v in enumerate(vals) if i not in indices_to_remove]
            for key, vals in train_data.items()
        }

        return Dataset.from_dict(filtered_train_data), Dataset.from_dict(test_data)

    elif choice == "ag-pubmed":
        ag_train_data, ag_test_data = load_bert_dataset_dicts("ag_news")
        pubmed_data = load_bert_dataset_dicts("pubmed")

        combined_test = {
            "text": ag_test_data["text"] + pubmed_data["text"],
            "origin": ag_test_data["origin"] + pubmed_data["origin"],
            "label": ag_test_data["label"] + pubmed_data["label"],
        }

        return Dataset.from_dict(ag_train_data), Dataset.from_dict(combined_test)

    elif choice == "mmlu":
        cs_data_val, cs_data_test, phil_data = load_bert_dataset_dicts("mmlu")

        combined_test = {
            "text": cs_data_test["text"] + phil_data["text"],
            "origin": cs_data_test["origin"] + phil_data["origin"],
            "label": cs_data_test["label"] + phil_data["label"],
        }

        return Dataset.from_dict(cs_data_val), Dataset.from_dict(combined_test)

    elif choice == "scienceqa-legalqa":
        scienceqa_train_data, scienceqa_test_data = load_bert_dataset_dicts("scienceqa")
        legalqa_data = load_bert_dataset_dicts("legalqa")

        train_indices_to_remove = {
            i for i, v in enumerate(scienceqa_train_data["label"]) if v != "geography"
        }

        filtered_train_data = {
            key: [v for i, v in enumerate(vals) if i not in train_indices_to_remove]
            for key, vals in scienceqa_train_data.items()
        }

        test_indices_to_remove = {
            i for i, v in enumerate(scienceqa_test_data["label"]) if v != "geography"
        }

        filtered_test_data = {
            key: [v for i, v in enumerate(vals) if i not in test_indices_to_remove]
            for key, vals in scienceqa_test_data.items()
        }

        combined_test = {
            "text": filtered_test_data["text"] + legalqa_data["text"],
            "origin": filtered_test_data["origin"] + legalqa_data["origin"],
            "label": filtered_test_data["label"] + legalqa_data["label"],
        }

        return Dataset.from_dict(filtered_train_data), Dataset.from_dict(combined_test)

    elif choice == "scienceqa":
        scienceqa_train_data, scienceqa_test_data = load_bert_dataset_dicts("scienceqa")

        indices_to_remove = {
            i for i, v in enumerate(scienceqa_train_data["label"]) if v != "geography"
        }

        filtered_train_data = {
            key: [v for i, v in enumerate(vals) if i not in indices_to_remove]
            for key, vals in scienceqa_train_data.items()
        }

        return Dataset.from_dict(filtered_train_data), Dataset.from_dict(
            scienceqa_test_data
        )

    else:
        raise ValueError(f"Dataset {choice} not supported.")


@lru_cache(maxsize=10000)
def cached_translate(text, source, target):
    if source == "zh":
        source = "zh-CN"
    if target == "zh":
        target = "zh-CN"
    return GoogleTranslator(source=source, target=target).translate(text)


def get_german_synonyms(word):
    try:
        url = "https://www.openthesaurus.de/synonyme/search"
        params = {"q": word, "format": "application/json"}
        response = requests.get(url, params=params)
        data = response.json()
        synonyms = set()
        for synset in data.get("synsets", []):
            for term in synset.get("terms", []):
                if term["term"].lower() != word.lower():
                    synonyms.add(term["term"])
        return list(synonyms) if synonyms else None
    except Exception as e:
        print(f"Error fetching German synonyms: {e}")
        return None


def get_omw_synonyms(word, lang):
    omw_lang = ISO_639_1_TO_3.get(lang)
    if not omw_lang:
        return None
    try:
        synsets = wordnet.synsets(word, lang=omw_lang)
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas(lang=omw_lang):
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms) if synonyms else None
    except:
        return None


def get_synonym(word, lang="en", tokenizer=None):
    supported_languages = ["en", "de", "es", "fr", "it", "ko", "pt", "ru", "zh"]

    if lang not in supported_languages:
        raise ValueError(f"Unsupported language: {lang}")

    try:
        # Step 1: Try native synonym lookup
        if lang == "de":
            synonyms = get_german_synonyms(word)
        else:
            synonyms = get_omw_synonyms(word, lang)

        # Step 2: If no native synonyms, fall back to English-based method
        if not synonyms:
            word_en = word if lang == "en" else cached_translate(word, lang, "en")
            synsets = wordnet.synsets(word_en)
            synonym_candidates = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if synonym.lower() != word_en.lower():
                        synonym_candidates.add(synonym)

            if not synonym_candidates:
                return None

            if tokenizer:
                valid_synonyms = []
                random.shuffle(list(synonym_candidates))
                for syn in synonym_candidates:
                    if lang == "en":
                        if len(tokenizer.tokenize(syn)) == 1:
                            valid_synonyms.append(syn)
                    else:
                        translated = cached_translate(syn, "en", lang)
                        if len(tokenizer.tokenize(translated)) == 1:
                            valid_synonyms.append(syn)
                if not valid_synonyms:
                    return None
                chosen_syn = random.choice(valid_synonyms)
            else:
                chosen_syn = random.choice(list(synonym_candidates))

            return (
                chosen_syn if lang == "en" else cached_translate(chosen_syn, "en", lang)
            )

        # Step 3: If we do have native synonyms, filter if tokenizer provided
        if tokenizer:
            filtered = [s for s in synonyms if len(tokenizer.tokenize(s)) == 1]
            if not filtered:
                return None
            return random.choice(filtered)

        return random.choice(synonyms)

    except Exception as e:
        print(f"Error: {e}")
        return None


def token_to_word(token, tokenizer):
    return tokenizer.decode([token]).strip()


def replace_tokens_with_synonyms(
    inputs, tokenizer, device, lang="en", replacement_prob=0.15
):
    stop_words = set(stopwords.words("english"))

    input_ids = inputs["input_ids"].clone()

    for i in range(input_ids.shape[0]):
        for j in range(input_ids.shape[1]):
            if random.random() < replacement_prob:
                token_id = input_ids[i, j].item()
                word = token_to_word(token_id, tokenizer)

                if (
                    word.lower() in stop_words
                    or word.startswith("##")
                    or not word.isalpha()
                ):
                    continue

                synonym = get_synonym(word, lang=lang, tokenizer=tokenizer)
                if not synonym:
                    synonym = word

                synonym_tokens = tokenizer(
                    synonym, return_tensors="pt", add_special_tokens=False
                ).to(device)

                if synonym_tokens["input_ids"].shape[1] == 1:
                    if synonym_tokens["input_ids"][0, 0] != token_id:
                        input_ids[i, j] = synonym_tokens["input_ids"][0, 0]

    return input_ids


def replace_tokens_with_random_tokens(inputs, tokenizer, device, replacement_prob=0.15):
    stop_words = set(stopwords.words("english"))

    input_ids = inputs["input_ids"].clone()

    vocab_size = tokenizer.vocab_size

    for i in range(input_ids.shape[0]):
        for j in range(input_ids.shape[1]):
            if random.random() < replacement_prob:
                token_id = input_ids[i, j].item()
                word = token_to_word(token_id, tokenizer)

                if (
                    word.lower() in stop_words
                    or word.startswith("##")
                    or not word.isalpha()
                ):
                    continue

                random_token_id = random.randint(0, vocab_size - 1)

                while random_token_id in tokenizer.all_special_ids:
                    random_token_id = random.randint(0, vocab_size - 1)

                input_ids[i, j] = random_token_id

    return input_ids


def load_multilingual_datasets(choice="finenews"):
    if choice == "finenews":
        languages = [
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

        sample_size = 100

        dataset = load_dataset(
            "maxidl/FineNews-unfiltered", name="CC-NEWS-2024-05", streaming=True
        )

        data = {
            "text": [],
            "language": [],
        }

        for lang in languages:
            lang_dataset = dataset[lang]
            content_samples = []
            count = 0

            for example in lang_dataset:
                if count >= sample_size:
                    break
                content_samples.append(example["text"])
                count += 1

            data["text"].extend(content_samples)
            data["language"].extend([lang] * len(content_samples))

        return Dataset.from_dict(data)

    elif choice == "mlsum":
        languages = [
            "de",
            "es",
            "fr",
            "ru",
        ]

        sample_size = 100

        data = {
            "text": [],
            "language": [],
        }

        for lang in languages:
            lang_dataset = load_dataset(
                "reciTAL/mlsum", name=lang, streaming=True, trust_remote_code=True
            )["train"]
            content_samples = []
            count = 0

            for example in lang_dataset:
                if count >= sample_size:
                    break
                content_samples.append(example["text"])
                count += 1

            data["text"].extend(content_samples)
            data["language"].extend([lang] * len(content_samples))

        return Dataset.from_dict(data)

    elif choice == "vript":
        languages = {
            "english": "en",
            "german": "de",
            "spanish": "es",
            "french": "fr",
            "italian": "it",
            "korean": "ko",
            "portuguese": "pt",
            "russian": "ru",
            "chinese": "zh",
        }

        sample_size = 100
        total_sample_size = len(languages) * sample_size

        count = 0

        dataset = load_dataset(
            "Mutonix/Vript_Multilingual", split="train", streaming=True
        )

        data = {
            languages[lang]: {
                "text": [],
                "language": [],
            }
            for lang in languages
        }

        for example in dataset:
            if example["lang"] not in languages:
                continue
            if count >= total_sample_size:
                break
            cur_lang = languages[example["lang"]]
            if len(data[cur_lang]["text"]) < sample_size:
                data[cur_lang]["text"].append(example["caption"]["content"])
                data[cur_lang]["language"].append(cur_lang)
                count += 1

        full_data = {
            "text": [],
            "language": [],
        }
        for lang in data:
            full_data["text"].extend(data[lang]["text"])
            full_data["language"].extend(data[lang]["language"])

        return Dataset.from_dict(full_data)

    elif choice == "commoncorpus":
        languages = {
            "English": "en",
            "German": "de",
            "Spanish": "es",
            "French": "fr",
            "Italian": "it",
            "Polish": "pl",
        }

        sample_size = 100
        total_sample_size = len(languages) * sample_size

        count = 0

        dataset = load_dataset("PleIAs/common_corpus", split="train", streaming=True)

        data = {
            languages[lang]: {
                "text": [],
                "language": [],
            }
            for lang in languages
        }

        for example in dataset:
            if example["language"] not in languages:
                continue
            if count >= total_sample_size:
                break
            cur_lang = languages[example["language"]]
            if len(data[cur_lang]["text"]) < sample_size:
                data[cur_lang]["text"].append(example["text"])
                data[cur_lang]["language"].append(cur_lang)
                count += 1

        full_data = {
            "text": [],
            "language": [],
        }
        for lang in data:
            full_data["text"].extend(data[lang]["text"])
            full_data["language"].extend(data[lang]["language"])

        return Dataset.from_dict(full_data)

    else:
        raise ValueError(f"Dataset {choice} not supported.")


def get_summarization_instruction(language="en"):
    if language == "en":
        return "Summarize the following text:\n"
    elif language == "de":
        return "Fassen Sie den folgenden Text zusammen:\n"
    elif language == "es":
        return "Resume el siguiente texto:\n"
    elif language == "fr":
        return "Résumez le texte suivant:\n"
    elif language == "it":
        return "Riassumi il seguente testo:\n"
    elif language == "ko":
        return "다음 텍스트를 요약하세요:\n"
    elif language == "pt":
        return "Resuma o seguinte texto:\n"
    elif language == "ru":
        return "Подведите итог следующему тексту:\n"
    elif language == "zh":
        return "总结以下文本：\n"
    else:
        raise ValueError(f"Language {language} not supported.")


def load_domain_specific_datasets(choice="ag-pubmed"):
    if choice == "ag-pubmed":
        ag_dataset = load_dataset("fancyzhx/ag_news", split="test")
        pubmed_dataset = load_dataset("MedRAG/pubmed", streaming=True)["train"]

        ag_labels = ag_dataset.features["label"].names

        sample_size = 200
        ag_sample_size = len(ag_labels) * sample_size

        count = 0

        data = {
            label: {
                "text": [],
                "label": [],
            }
            for label in ag_labels + ["Medicine"]
        }

        for example in ag_dataset:
            if count >= ag_sample_size:
                break
            cur_label = ag_labels[example["label"]]
            if len(data[cur_label]["text"]) < sample_size:
                data[cur_label]["text"].append(example["text"])
                data[cur_label]["label"].append(cur_label)
                count += 1

        pubmed_samples = []
        count = 0

        for example in pubmed_dataset:
            if count >= sample_size:
                break
            data["Medicine"]["text"].append(example["content"])
            data["Medicine"]["label"].append("Medicine")
            count += 1

        full_data = {
            "text": [],
            "label": [],
        }
        for label in data:
            full_data["text"].extend(data[label]["text"])
            full_data["label"].extend(data[label]["label"])

        return Dataset.from_dict(full_data)

    else:
        raise ValueError(f"Dataset {choice} not supported.")
