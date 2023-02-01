import codecs
import pickle
import difflib

from torch.utils.data import Dataset

class ProcessKnowledgeNet(Dataset):
    r"""PyTorch Dataset class for loading data.

    This is where the data parsing happens.

    This class is built for reusability.

    Arguments:

      path (:obj:`str`):
          Path to the data partition.

    """
    def __init__(self, split, use_tokenizer):

        with codecs.open('/path/to/basic_corpus_fold_1_nomarkup_complete_annotation.pickle', "rb") as input_file:
            data_dict1 = pickle.load(input_file)

        with codecs.open('/path/to/basic_corpus_fold_2_nomarkup_complete_annotation.pickle', "rb") as input_file:
            data_dict2 = pickle.load(input_file)

        with codecs.open('/path/to/basic_corpus_fold_3_nomarkup_complete_annotation.pickle', "rb") as input_file:
            data_dict3 = pickle.load(input_file)

        with codecs.open('/path/to/basic_corpus_fold_4_nomarkup_complete_annotation.pickle', "rb") as input_file:
            data_dict4 = pickle.load(input_file)


        data_dict = {}
        data_dict.update(data_dict1)
        data_dict.update(data_dict2)
        data_dict.update(data_dict3)
        data_dict.update(data_dict4)

        if split == 'train':

            with codecs.open('/path/to/fold_1_annotated_facts.pickle', "rb") as input_file:
                kn_dict1 = pickle.load(input_file)

            with codecs.open('/path/to/fold_2_annotated_facts.pickle', "rb") as input_file:
                kn_dict2 = pickle.load(input_file)

            #with codecs.open('/path/to/fold_3_annotated_facts.pickle', "rb") as input_file:
             #   kn_dict3 = pickle.load(input_file)

            with codecs.open('/path/to/fold_4_annotated_facts.pickle', "rb") as input_file:
                kn_dict4 = pickle.load(input_file)

            kn_dict = {}
            kn_dict.update(kn_dict1)
            kn_dict.update(kn_dict2)
            # kn_dict.update(kn_dict3)
            kn_dict.update(kn_dict4)

        if split == 'test':

            #with codecs.open('/path/to/fold_1_annotated_facts.pickle', "rb") as input_file:
             #   kn_dict1 = pickle.load(input_file)

            #with codecs.open('/path/to/fold_2_annotated_facts.pickle', "rb") as input_file:
             #   kn_dict2 = pickle.load(input_file)

            with codecs.open('/path/to/fold_3_annotated_facts.pickle', "rb") as input_file:
                kn_dict3 = pickle.load(input_file)

            #with codecs.open('/path/to/fold_4_annotated_facts.pickle', "rb") as input_file:
             #   kn_dict4 = pickle.load(input_file)

            kn_dict = {}
            # kn_dict.update(kn_dict1)
            # kn_dict.update(kn_dict2)
            kn_dict.update(kn_dict3)
            # kn_dict.update(kn_dict4)

        KN_keys = list(kn_dict.keys())
        our_keys = list(data_dict.keys())

        facts_labels = []
        train_labels = []
        train_lines = []
        labels = []
        counter = 0
        for KN_id in KN_keys:
            KN_doc_data = kn_dict.get(KN_id)
            for our_id in our_keys:
                our_doc_data = data_dict.get(our_id)
                # Document level, we loop over the all the sentences in the document
                for our_index, our_propos in enumerate(our_doc_data.keys()):
                    if our_doc_data.get(our_index) is not None:
                        our_sentence = our_doc_data.get(our_index).sentence
                        our_props_list = our_doc_data.get(our_index).props
                    for KN_index, KN_propos in enumerate(KN_doc_data):
                        if KN_doc_data[KN_index] is not None:
                            KN_sentence = KN_doc_data[KN_index].sentence
                            KN_props_list = KN_doc_data[KN_index].props
                            if our_sentence == KN_sentence:
                                #Sentence level, we loop over all the clauses on the sentence
                                for KN_k, KN_prop in enumerate(KN_props_list):
                                    KN_prop_members = KN_prop.members
                                    KN_fact = build_fact(KN_prop)
                                    KN_fact = KN_prop.members['subj'].text + ' ' + KN_prop.members['obj'].text
                                    KN_label = KN_prop.members['predicate'].text

                                    for our_k, our_prop in enumerate(our_props_list):

                                        our_prop_members = our_prop.members
                                        our_fact = build_fact(our_prop)
                                        fact = our_prop.members['subj'].text + ' ' + our_prop.members['obj'].text

                                        similarity = difflib.SequenceMatcher(None, fact, KN_fact)
                                        if similarity.ratio() >= 0.9:

                                            train_lines.append(our_fact)
                                            train_labels.append(KN_label)


        self.texts = []
        self.labels = []
        # Since the labels are defined by folders with data we loop
        # through each label.

        # Save content.
        self.texts = train_lines
        # Save encode labels.
        self.labels = train_labels

        # Number of exmaples.
        self.n_examples = len(self.labels)

        return

    def __len__(self):
        r"""When used `len` return the number of examples.

        """

        return self.n_examples

    def __getitem__(self, item):
        r"""Given an index return an example from the position.

        Arguments:

          item (:obj:`int`):
              Index position to pick an example to return.

        Returns:
          :obj:`Dict[str, str]`: Dictionary of inputs that contain text and
          asociated labels.

        """

        return {'text' :self.texts[item],
                'label' :self.labels[item]}


def build_fact(prop):
    r"""Function to build back the fact for alignement with the ground truth.

    Arguments:

      item (:obj:`PropMember`):
          PropMember object storing the clause.

    Returns:
      :obj:`string`: String with the entire clause as a single string
      asociated labels.

    """

    return prop.members['subj'].text + ' ' + prop.members['predicate'].text + ' ' + prop.members['obj'].text
