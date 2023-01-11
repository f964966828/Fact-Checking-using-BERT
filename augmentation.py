import json
import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

first_model_name = 'Helsinki-NLP/opus-mt-en-fr'
first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
first_model = MarianMTModel.from_pretrained(first_model_name)
first_model = first_model.to('cuda')

second_model_name = 'Helsinki-NLP/opus-mt-fr-en'
second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
second_model = MarianMTModel.from_pretrained(second_model_name)
second_model = second_model.to('cuda')

def format_text(language_code, text):
    return f">>{language_code}<< {text}"


def perform_translation(text, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_text = format_text(language, text)
    
    encoding = tokenizer(formated_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    translated = model.generate(**encoding.to('cuda'))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

def backtranslation(original_text):
    translated_text = perform_translation(original_text, first_model, first_model_tkn)
    back_translated_text = perform_translation(translated_text, second_model, second_model_tkn)

    return back_translated_text

with open('./2022-inlp-final/train.json') as f:
    train_data = json.load(f)
with open('./2022-inlp-final/train_evidence.json') as f:
    train_data_evidence = json.load(f)

labels = list()
for data in train_data:
    label = data['label']['rating']
    labels.append(label)

print('Origin Distribution:')
print(pd.Series(labels).value_counts())

text = "Back translation: translate back each of those translated data into the original language, meaning a translation from French to English."
result = backtranslation(text)
print("----Example of backtranslation----")
print(f"Before: {text}")
print(f"After: {result}")

augment_data = {}
progress = tqdm(total = (pd.Series(labels) == 2).sum())
for data in train_data:
    claim = data['metadata']['claim']
    label = data['label']['rating']
    file_id = str(data['metadata']['id'])

    # only label = 2 need to do augmentation
    if label == 2:
        claim_result = backtranslation(claim)

        sentences = list()        
        for evidence in train_data_evidence[file_id]:
            evidence_result = backtranslation(evidence)
            sentences.append(evidence_result)

        aug_data = {}
        aug_data['claim'] = claim_result
        aug_data['label'] = label
        aug_data['sentences'] = sentences
        augment_data[file_id] = aug_data 

        progress.update(1)


with open('./2022-inlp-final/aug_evidence.json', 'w', encoding='utf-8') as fout:
    json.dump(augment_data, fout, ensure_ascii=False, indent=4)
