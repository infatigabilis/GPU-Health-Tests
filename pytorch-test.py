import sys

from nlp import load_dataset
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments

from config import EPOCHS, BATCH_SIZE

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


def compute_accuracy(eval_prediction):
    preds = eval_prediction.predictions.argmax(axis=1)
    acc = accuracy_score(eval_prediction.label_ids, preds)
    return {'accuracy': acc}


train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_accuracy
)

trainer.train()

final_acc = trainer.evaluate()['eval_accuracy']

if final_acc > 0.9:
    print('Test passed')
else:
    print("Test failed: The model hasn't reached the expected metrics. Perhaps GPU's computational ability is broken (for example, due to overheating)", file=sys.stderr)
