from transformers import TFBertForSequenceClassification, BertTokenizerFast, TFTrainer, TFTrainingArguments
import tensorflow as tf
from nlp import load_dataset
from config import *

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_dataset, test_dataset = load_dataset('imdb', split=['train[:10%]', 'test[:10%]'])
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
train_dataset.set_format('tensorflow', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('tensorflow', columns=['input_ids', 'attention_mask', 'label'])

def to_tfds(dataset):
    x = {}
    x['input_ids'] = train_dataset['input_ids'].to_tensor(default_value=0, shape=[None, tokenizer.max_len])
    x['attention_mask'] = train_dataset['attention_mask'].to_tensor(default_value=0, shape=[None, tokenizer.max_len])
    y = train_dataset['label']
    return tf.data.Dataset.from_tensor_slices((x, y))

tf_train_dataset = to_tfds(train_dataset)
tf_test_dataset = to_tfds(test_dataset)

training_args = TFTrainingArguments(
    output_dir='./results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs',
)

with training_args.strategy.scope():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=tf_train_dataset,
    eval_dataset=tf_test_dataset
)

trainer.train()
