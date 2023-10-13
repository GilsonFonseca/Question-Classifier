# -*- coding: utf-8 -*-

from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW, BertConfig, TextClassificationPipeline, logging
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, matthews_corrcoef
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import files_ms_client
import seaborn as sn
import pandas as pd
import numpy as np
import datetime
import torch
import time
import os


logging.set_verbosity_error()


# GPU
device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
  print('GPU encontrada em: {}'.format(device_name))
else:
  print('GPU não encontrada')

# Set enviroment configs up. Default values can be changed by altering
# the second argument in each "get" call
FILES_SERVER = os.environ.get("FILES_SERVER", "200.17.70.211:10162")

def myCode(msg):
  FILE = 'basebloom.xlsx'

  files_ms_client.download(msg["file"]["name"], FILE, url="http://" + FILES_SERVER)
  df = pd.read_excel(FILE)
  df.sample(frac=1)

  # Remove os dados N/A da base, assim como tabelas não usadas
  df = df.dropna()

  # Recebe df
  dataset = df.copy()

  num_classes = len(df["Level"].value_counts())

  dataset['category_id'] = df['Level'].factorize()[0]
  category_id_df = dataset[['Level', 'category_id']].drop_duplicates().sort_values('category_id')
  category_to_id = dict(category_id_df.values)
  id_to_category = dict(category_id_df[['category_id', 'Level']].values)


  dataset['Labels'] = dataset['Level'].map({'Knowledge': 0,
                                              'Comprehension': 1,
                                              'Application': 2,
                                              'Analysis': 3,
                                              'Synthesis': 4,
                                              'Evalution': 5})

  # # drop unused column
  dataset = dataset.drop(["Level"], axis=1)
  dataset = dataset.drop(["category_id"], axis=1)

  dataset['Question'] = dataset['Question'].str.lower()

  datasetTest = dataset.groupby('Labels').sample(n=17)
  datasetTest.to_csv('datasetTest.csv', index=False)
  dataset = dataset.drop_duplicates(subset="Question")
  for row, data in enumerate(dataset.values):
    for dup in datasetTest.values:
      if((data == dup).all()):
        dataset = dataset[dataset.Question != data[0]]

  # Recebe a lista com as sentenças e os labels
  sentences = dataset.Question.values
  labels = dataset.Labels.values

  tokenizer = BertTokenizer.from_pretrained(
      'bert-base-uncased',
      # 'bert-large-uncased',
      do_lower_case = True
      )
  tokenizer.save_pretrained('./saved_model/')

  input_ids = []
  attention_masks = []

  def preprocessing(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
      - input_ids: list of token ids
      - token_type_ids: list of token type ids
      - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
                          input_text,
                          add_special_tokens = True,
                          truncation=True,
                          max_length = 32,
                          padding='max_length',
                          return_attention_mask = True,
                          return_tensors = 'pt'
                    )


  for sample in sentences:
    encoding_dict = preprocessing(sample, tokenizer)
    input_ids.append(encoding_dict['input_ids'])
    attention_masks.append(encoding_dict['attention_mask'])


  input_ids = torch.cat(input_ids, dim = 0)
  attention_masks = torch.cat(attention_masks, dim = 0)
  labels = torch.tensor(labels)

  def format_time(elapsed):
      '''
      Takes a time in seconds and returns a string hh:mm:ss
      '''
      # Round to the nearest second.
      elapsed_rounded = int(round((elapsed)))

      # Format as hh:mm:ss
      return str(datetime.timedelta(seconds=elapsed_rounded))

  # Function to calculate the accuracy of our predictions vs labels
  def flat_accuracy(preds, labels):
      pred_flat = np.argmax(preds, axis=1).flatten()
      labels_flat = labels.flatten()
      return np.sum(pred_flat == labels_flat) / len(labels_flat)

  """TREINAMENTO

  """

  # Function to train the model
  def bert_training(model, optimizer, train_dataloader, device, scheduler, n_Train, epoch_i):
      print("")
      print('======== Train {:} Epoch {:} / {:} ========'.format(n_Train+1, epoch_i + 1, epochs))
      print('Training...')
      t0 = time.time()
      # Reset the total loss for this epoch.
      total_loss = 0
      # Set model to training mode
      model.train()
      # For each batch of training data...
      for step, batch in enumerate(train_dataloader):
          # Progress update every 40 batches.
          if step % 40 == 0 and not step == 0:
              # Calculate elapsed time in minutes.
              elapsed = format_time(time.time() - t0)
              # Report progress.
              print('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
          # `batch` contains three pytorch tensors:
          #   [0]: input ids
          #   [1]: attention masks
          #   [2]: labels
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = torch.nn.functional.one_hot(batch[2].to(device).detach().clone(), num_classes=num_classes).to(
              torch.float
          )
          # Always clear any previously calculated gradients before performing a
          # backward pass.
          model.zero_grad()

          outputs = model(b_input_ids,
                      token_type_ids=None,
                      attention_mask=b_input_mask,
                      labels=b_labels)

          loss = outputs[0]
          # `loss` is a Tensor containing a single value; the `.item()` function just returns the Python value from the tensor.
          total_loss += loss.item()
          # Perform a backward pass to calculate the gradients.
          loss.backward()
          # Clip the norm of the gradients to 1.0.
          # This is to help prevent the "exploding gradients" problem.
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          # Update parameters and take a step using the computed gradient.
          # The optimizer dictates the "update rule"--how the parameters are
          # modified based on their gradients, the learning rate, etc.
          optimizer.step()
          # Update the learning rate.
          scheduler.step()
      # Calculate the average loss over the training data.
      print("")
      print("  Average training loss: {0:.2f}".format(total_loss / len(train_dataloader)))
      print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

  # Function to evaluate the model
  def bert_evaluating(model, validation_dataloader, device):
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")
        t0 = time.time()
        # accuracy = bert_evaluating(validation_dataloader, device)
        # Report the final accuracy for this validation run.
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
        # Tracking variables
        eval_accuracy = 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy
            # Track the number of batches
            nb_eval_steps += 1
            # return eval_accuracy/nb_eval_steps
        print("")
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        return eval_accuracy/nb_eval_steps

  def config_model():
    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained(
      "bert-base-uncased",
      # "bert-large-uncased",
      num_labels = num_classes, # The number of output labels--2 for binary classification.
                      # You can increase this for multi-class tasks.
      output_attentions = False, # Whether the model returns attentions weights.
      output_hidden_states = False, # Whether the model returns all hidden-states.
      problem_type="multi_label_classification",
      id2label={0 : 'Knowledge', 1 : 'Comprehension', 2 : 'Application', 3 : 'Analysis', 4 : 'Synthesis', 5 : 'Evalution'}
    )

    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = 5e-5,
                                  eps = 1e-08
                                  )
    model.save_pretrained('./saved_model/')
    # Run on GPU
    model.cuda()

    return model, optimizer

  def dataloader(X_train, y_train, X_val, y_val):

    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.1)

    if(len(X_train) == len(y_train) and len(X_train) != len(train_masks)):
      X_train = torch.cat([X_train[0:len(X_train)-1], X_train[len(X_train):]])
      y_train = torch.cat([y_train[0:len(y_train)-1], y_train[len(y_train):]])

    if(len(X_val) == len(y_val) and len(X_val) != len(validation_masks)):
      validation_masks = torch.cat([validation_masks[0:len(validation_masks)-1], validation_masks[len(validation_masks):]])

    train_data = [X_train, y_train, train_masks]

    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    train_inputs = train_data[0].detach().clone()
    validation_inputs = X_val.detach().clone()
    train_labels = train_data[1].detach().clone()
    validation_labels = y_val.detach().clone()
    train_masks = train_data[2].detach().clone()
    validation_masks = validation_masks.detach().clone()

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader, validation_inputs, validation_masks, validation_labels

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # recommend between 2 and 4
  epochs = 8
  # recommend 16 or 32.
  batch_size = 16

  skf = StratifiedKFold(n_splits=10, shuffle = True)
  X, y = input_ids, labels
  resultArray = []

  for n_Train, (train_index, test_index) in enumerate(skf.split(X, y)):

    model, optimizer = config_model()

    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

    train_dataloader, validation_dataloader, validation_inputs, validation_masks, validation_labels = dataloader(X_train, y_train, X_val, y_val)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        bert_training(model, optimizer, train_dataloader, device, scheduler, n_Train, epoch_i)

        # ========================================
        #               Validation
        # ========================================

        accuracy = bert_evaluating(model, validation_dataloader, device)

    resultArray.append(accuracy)
    print("")
    print("Training complete!")

  print(f"Cross {resultArray}")
  print(f"Média {np.mean(resultArray)}")
  print(f"Desvio Padrão {np.std(resultArray)}")

  test_ids = []
  test_masks = []

  for t_sample in datasetTest.Question.values:
    t_encoding_dict = preprocessing(t_sample, tokenizer)
    test_ids.append(t_encoding_dict['input_ids'])
    test_masks.append(encoding_dict['attention_mask'])


  test_ids = torch.cat(test_ids, dim = 0)
  test_masks = torch.cat(test_masks, dim = 0)
  test_labels = torch.tensor(datasetTest.Labels.values)

  # Create attention masks
  attention_masks = []
  # Create a mask of 1s for each token followed by 0s for padding
  for seq in test_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

  # Convert to tensors.
  prediction_inputs = test_ids.detach().clone()
  prediction_masks = test_masks.detach().clone()
  prediction_labels = test_labels.detach().clone()
  # Set the batch size.
  batch_size = 32
  # Create the DataLoader.
  prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

  # Prediction on test set
  print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
  # Put model in evaluation mode
  model.eval()
  # Tracking variables
  predictions , true_labels = [], []
  # Predict
  for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)
  print('DONE.')

  matthews_set = []
  # Evaluate each test batch using Matthew's correlation coefficient
  print('Calculating Matthews Corr. Coef. for each batch...')
  # For each input batch...
  for i in range(len(true_labels)):

    # The predictions for this batch are a 2-column ndarray (one column for "0"
    # and one column for "1"). Pick the label with the highest value and turn this
    # in to a list of 0s and 1s.
    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

    # Calculate and store the coef for this batch.
    matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
    matthews_set.append(matthews)

  matthews_set

  # Combine the predictions for each batch into a single list of 0s and 1s.
  flat_predictions = [item for sublist in predictions for item in sublist]
  flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
  # Combine the correct labels for each batch into a single list.
  flat_true_labels = [item for sublist in true_labels for item in sublist]
  # Calculate the MCC
  mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
  cm = confusion_matrix(flat_true_labels, flat_predictions)
  print(f"Micro {precision_recall_fscore_support(flat_true_labels, flat_predictions, average='micro')}") # y_true, são as classificações corretas da base de teste, e y_test os resultados da predição do MNB
  print(f"Macro {precision_recall_fscore_support(flat_true_labels, flat_predictions, average='macro')}")
  print(f"Weighted {precision_recall_fscore_support(flat_true_labels, flat_predictions, average='weighted')}")

  print('MCC: %.3f' % mcc)
  print(cm)
  print(metrics.classification_report(flat_true_labels, flat_predictions, labels=[0,1,2,3,4,5]))


  df_cm = pd.DataFrame(cm, range(num_classes), range(num_classes))
  plt.figure(figsize=(11,8))
  sn.set(font_scale=1.4) # for label size
  sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

  plt.show()
  graph = plt.savefig("graph.png")
 
# Retorno pipeline
  msg["qc-image"] = files_ms_client.upload("graph.png", url="http://" + FILES_SERVER)



  tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english') # Identificar dentre as falso positivas as mais raras
  features = tfidf.fit_transform(sentences).toarray()
  labels = labels
  features.shape
  print(features.size)
  print(flat_predictions.size)
  sentences[:10]

  # bloom = ('Knowledge', 'Comprehension', 'Application', 'Analysis', 'Synthesis', 'Evalution')
  # pipe = TextClassificationPipeline(model = model, tokenizer=tokenizer, top_k=None)
  pipe = TextClassificationPipeline(model = model, tokenizer=tokenizer)
  pipe.device = torch.device('cuda:0')

  pipe(["Solve the following problem",
        "differentiate the block diagram of the central processing unit",
        "Create a new ordinal positions",
        "define four types of traceability"])

  pipe = TextClassificationPipeline(model = model, tokenizer=tokenizer, top_k=None)
  # pipe = TextClassificationPipeline(model = model, tokenizer=tokenizer)
  pipe.device = torch.device('cuda:0')

  # bloom = ('Knowledge', 'Comprehension', 'Application', 'Analysis', 'Synthesis', 'Evalution')
  pipe("demonstrate the block diagram of the central processing unit")

  # bloom = ('Knowledge', 'Comprehension', 'Application', 'Analysis', 'Synthesis', 'Evalution')
  # for id, sentence in enumerate(datasetTest.Question.values):
  #   if(flat_true_labels[id] != flat_predictions[id]): # Posso fazer isso, pois as sentenças estão na ordem
  #     # if(flat_true_labels[id] != flat_predictions[id] & flat_predictions[id] == 2):
  #       print(flat_true_labels[id], flat_predictions[id], sentence, pipe(sentence))

  return msg