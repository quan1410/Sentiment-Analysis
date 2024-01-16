!pip install datasets
from datasets import *
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
dataset = load_dataset("C:/Project/movie_reviews/movie_reviews.csv")
dataset
model = TFAutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Model's input Representation
inputs = tokenizer(
    ['This is the first sentence of the list', 'And this is the second sentence'],
    padding='max_length',
    truncation=True,
    return_tensors='tf'
)
inputs
# Model's output Representation
output = model(inputs)
output
train_testvalid = dataset['train'].train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
ds = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']}
)
ds
def tokenize(batch):
  return tokenizer(batch['text'], padding='max_length', truncation=True)
ds_encoded = ds.map(tokenize, batched=True, batch_size=None)
ds_encoded
ds_encoded.set_format(
    'tf', # Convert these to tensor
    columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
)

batch_size = 8

def order(inputs):
  data = list(inputs.values())
  return {
      'input_ids': data[1],
      'attention_mask': data[2],
      'token_type_ids': data[3]
  }, data[0]
train_dataset = tf.data.Dataset.from_tensor_slices(ds_encoded['train'][:])
train_dataset = train_dataset.batch(batch_size).shuffle(1000)
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)
eval_dataset = tf.data.Dataset.from_tensor_slices(ds_encoded['valid'][:])
eval_dataset = eval_dataset.batch(batch_size)
eval_dataset = eval_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices(ds_encoded['test'][:])
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)
next(iter(train_dataset))
class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)
classifier = BERTForClassification(model, num_classes=2)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
import os
checkpoint_path = "checkpoint_/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
history = classifier.fit(
    train_dataset,
    validation_data=eval_dataset,
    epochs=3,
    callbacks=[cp_callback]
)
from google.colab import drive
drive.mount('/content/drive')
classifier.save('my_model.keras')
classifier.evaluate(test_dataset)
def predict_sentiment(text):
  # Text the sentence -> Process it to be BERT's input -> convert the inputs to tensor -> call model
  inputs = tokenizer(
    [text],
    padding='max_length',
    truncation=True,
    return_tensors='tf'
  )
  output = classifier.call(inputs).numpy()[0]
  label = np.argmax(output)
  confidence = output[label]
  return label, confidence

sentence = "This is a test sentence which feels very good."
label, confidence = predict_sentiment(sentence)
print(f"Sentence: {sentence}")
print(f"Label: {label} ({'Negative' if label == 0 else 'Positive'}) with confidence score {confidence: .2f}.")
# Check with real dataset
for i in range(10):
  text = dataset['train']['text'][i]
  real_label = dataset['train']['label'][i]
  predicted_label = predict_sentiment(text)[0]
  print(f"Sentence no.{i} has label {real_label} -> model predicted {predicted_label}.")
submission = pd.read_csv("test_data.csv")
submission.head(5)
submission['Category'] = submission['text'].apply(lambda x: predict_sentiment(x)[0])
submission.head(5)
submission = submission.drop('text', axis=1)
submission.to_csv('submission.csv', index=False)