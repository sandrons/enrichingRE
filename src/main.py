##### © Copyright 2023 ********** *********.

#Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          AutoConfig,
                          AutoTokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          AutoModelForSequenceClassification)

from classifier import Classifier


def load_train():
    with open('path/to/data/train.txt') as f:
        train_data = f.readlines()
    train_dataset = []
    for line in train_data:
        text, label = line.split('\t')
        train_dataset.append({'text': text, 'label': label.replace('\n', '')})
    return train_dataset


def load_test():
    with open('path/to/data/test.txt') as f:
        test_data = f.readlines()
    test_dataset = []
    for line in test_data:
        text, label = line.split('\t')
        test_dataset.append({'text': text, 'label': label.replace('\n', '')})

    return test_dataset

def REwithOpenIE():

    # Set seed for reproducibility.
    set_seed(123)

    # Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
    epochs = 3

    # Number of batches.
    batch_size = 16

    # Pad or truncate text sequences to a specific length
    # if `None` it will use maximum sequence of word piece tokens allowed by model.
    max_length = 128

    # Look for gpu to use. Will use `cpu` by default if no gpu found.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Name of transformers model - will use already pretrained model.
    # Path of transformer model - will load your own model from local disk.
    pretained = 'roberta-base'

    # Dictionary of labels and their id - this will be used to convert.
    # String labels to number ids.
    labels_ids = {'DATE_OF_BIRTH': 0, 'DATE_OF_DEATH': 1,
                  'PLACE_OF_RESIDENCE': 2, 'PLACE_OF_BIRTH': 3,
                  'NATIONALITY': 4, 'EMPLOYEE_OR_MEMBER_OF': 5,
                  'EDUCATED_AT': 6, 'POLITICAL_AFFILIATION': 7,
                  'CHILD_OF': 8, 'SPOUSE': 9,
                  'DATE_FOUNDED': 10, 'HEADQUARTERS': 11,
                  'SUBSIDIARY_OF': 12,
                  'FOUNDED_BY': 13, 'CEO': 14
                  }

    # This is used to decide size of classification head.
    n_labels = len(labels_ids)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(pretained)

    model_config = AutoConfig.from_pretrained(pretained, num_labels=n_labels)
    # Get the model.
    print('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(pretained, config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`' % device)
    classifier = Classifier(use_tokenizer=tokenizer,labels_encoder=labels_ids,max_sequence_len=max_length)

    train_dataset = load_train()
    test_dataset = load_test()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=classifier)
    valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=classifier)

    #AdamW is a class from the huggingface library (not pytorch)
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives us the number of batches.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,  num_training_steps=total_steps)

    def train(dataloader, optimizer_, scheduler_, device_, model):
        r"""
        Train pytorch model on a single pass through the data loader.

        This function is built for reusability: it can be used as is as long
          as the `dataloader` outputs a batch in dictionary format that can be passed
          straight into the model - `model(**batch)`.

        Arguments:

            dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
                Parsed data into batches of tensors.

            optimizer_ (:obj:`transformers.optimization.AdamW`):
                Optimizer used for training.

            scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
                PyTorch scheduler.

            device_ (:obj:`torch.device`):
                Device used to load tensors before feeding to model.

            model (:obj:`transformers.AutoModelForSequenceClassification`):
                Pretrained model from HuggingFace.

        Returns:

            :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
              Labels, Train Average Loss].
        """

        # Tracking variables.
        predictions_labels = []
        true_labels = []
        # Total loss for this epoch.
        total_loss = 0

        # Put the model into training mode.
        model.train()

        # For each batch of training data...
        for batch in dataloader:
            # Add original labels - use later for evaluation.
            true_labels += batch['labels'].numpy().flatten().tolist()

            # move batch to device
            batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

            # Always clear any previously calculated gradients before performing a
            # backward pass.
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this a bert model function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

            outputs = model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to calculate training accuracy.
            loss, logits = outputs[:2]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer_.step()

            # Update the learning rate.
            scheduler_.step()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()

        # Calculate the average loss over the training data.
        avg_epoch_loss = total_loss / len(dataloader)

        # Return all true labels and prediction for future evaluations.
        return true_labels, predictions_labels, avg_epoch_loss

    def validation(dataloader, device_, model):
        r"""Validation function to evaluate model performance on a
        separate set of data.

        This function will return the true and predicted labels so we can use later
        to evaluate the model's performance.

        This function is built with reusability in mind: it can be used as is as long
          as the `dataloader` outputs a batch in dictionary format that can be passed
          straight into the model - `model(**batch)`.

        Arguments:

          dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
                Parsed data into batches of tensors.

          device_ (:obj:`torch.device`):
                Device used to load tensors before feeding to model.

          model (:obj:`transformers.AutoModelForSequenceClassification`):
                Pretrained model from HuggingFace.

        Returns:

          :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
              Labels, Train Average Loss]
        """

        # Tracking variables
        predictions_labels = []
        true_labels = []
        # total loss for this epoch.
        total_loss = 0

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Evaluate data for one epoch
        for batch in dataloader:
            # add original labels
            true_labels += batch['labels'].numpy().flatten().tolist()

            # move batch to device
            batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

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

                outputs = model(**batch)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple along with the logits. We will use logits
                # later to to calculate training accuracy.
                loss, logits = outputs[:2]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # get predicitons to list
                predict_content = logits.argmax(axis=-1).flatten().tolist()

                # update list
                predictions_labels += predict_content

        # Calculate the average loss over the training data.
        avg_epoch_loss = total_loss / len(dataloader)

        # Return all true labels and prediction for future evaluations.
        return true_labels, predictions_labels, avg_epoch_loss

    # Loop through each epoch.
    print('Epoch')
    for epoch in range(epochs):
        print()
        print('Training on batches...')
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device, model)
        train_acc = accuracy_score(train_labels, train_predict)
        train_precision = precision_score(train_labels, train_predict, average='micro')
        train_recall = recall_score(train_labels, train_predict, average='micro')

        # Get prediction form model on validation data.
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(valid_dataloader, device, model)
        val_acc = accuracy_score(valid_labels, valid_predict)
        val_precision = precision_score(train_labels, train_predict, average='micro')
        val_recall = recall_score(train_labels, train_predict, average='micro')

        # Print loss and accuracy values to see how training evolves.
        print(
            "  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f - train_precision: %.5f - valid_precision: %.5f - train_recall: %.5f - valid_recall: %.5f" % (
            train_loss, val_loss, train_acc, val_acc, train_precision, val_precision, train_recall, val_recall))
        print()

        print("  train_loss: %.5f - train_acc: %.5f " % (train_loss, train_acc))
        print()

    # Get prediction form model on validation data. This is where you should use
    # your test data.
    true_labels, predictions_labels, avg_epoch_loss = validation(valid_dataloader, device, model)

    # Create the evaluation report.
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
    # Show the evaluation report.
    print(evaluation_report)

    print('ended succefully')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    REwithOpenIE()
