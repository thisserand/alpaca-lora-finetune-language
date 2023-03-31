## Fine-Tune Alpaca For Any Language
In this repository, I've collected all the sources I used to create the [YouTube video](https://youtu.be/yTROqe8T_eA) and the [Medium article](https://medium.com/p/370f63753f94) on fine-tuning the alpaca model for any language. You can find more details on how to do this in both articles. 

> Note: This repository is intended to provide additional material to the video. This means that you can't just clone this repository, run three commands, and the fine-tuning is done. This is mainly because the implementation of the Alpaca-LoRA repository is constantly being improved and changed, so it would be difficult to keep the Alpaca-LoRA repository files (which I have partially customized) up-to-date.

## Translation
Run each cell in the [translation notebook](./translation.ipynb) to translate the [cleaned dataset](https://github.com/gururise/AlpacaDataCleaned/blob/main/alpaca_data_cleaned.json) into your target language. To do this, make sure you configure your target language and set up your auth_key for the DeepL API or OpenAI API.

In this [file](./data/source_tasks/tasks_translated_en.json) you can see all the tasks I translated, and in this [file](./data/source_tasks/tasks_not_translated_en.json) you can see all the tasks from the original dataset that I did not translate.

And these are my translated data sets that I used to fine-tune the Alpaca model:

- [alpaca-lora-german-7b-deepl-12k](./data/translated/translated_tasks_de_deepl_12k.json)
- [alpaca-lora-german-7b-deepl-4k](./data/translated/translated_tasks_de_deepl_4k.json)
- [alpaca-lora-german-7b-openai-12k](./data/translated/translated_tasks_de_openai_12k.json)

Thanks to [@JSmithOner](https://github.com/JSmithOner) for translating the whole dataset (52k tasks) to German using the Google Translator:
- [tanslated_tasks_de_google_52k](./data/translated/tanslated_tasks_de_google_52k.json)

## Fine-Tuning
```
python finetune.py --base_model="decapoda-research/llama-7b-hf" --data-path "translated_task_de_deepl_12k.json"
```

## Evaluation
```
python generate_eval.py
```

You can see my evaluation results in this [file](./data/evaluation/evaluation_result_first_20.txt) or in my [Medium article](https://medium.com/p/8e363a0a99ca).


## Trained Models (Hugging Face)
- [alpaca-lora-german-7b-deepl-12k](https://huggingface.co/thisserand/alpaca-lora-german-7b-deepl-12k)
- [alpaca-lora-german-7b-deepl-4k](https://huggingface.co/thisserand/alpaca-lora-german-7b-deepl-4k)
- [alpaca-lora-german-7b-openai-12k](https://huggingface.co/thisserand/alpaca-lora-german-7b-openai-12k)
