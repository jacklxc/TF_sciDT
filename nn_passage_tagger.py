import warnings
import sys
import codecs
import numpy as np
import argparse
import json
import pickle

from rep_reader import RepReader
from util import read_passages, evaluate, make_folds

import keras.backend as K
from keras.activations import softmax
from keras.models import Sequential, model_from_json
from keras.layers import Input, LSTM, GRU, Dense, Dropout, TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.optimizers import Adam, RMSprop, SGD
from crf import CRF
from attention import TensorAttention
from keras_extensions import HigherOrderTimeDistributedDense

class PassageTagger(object):
    def __init__(self, word_rep_file=None, pickled_rep_reader=None):
        if pickled_rep_reader:
            self.rep_reader = pickled_rep_reader
        elif word_rep_file:
            self.rep_reader = RepReader(word_rep_file)
        try:
            self.input_size = self.rep_reader.rep_shape[0]
        except:
            self.input_size = 0
        self.tagger = None

    def make_data(self, trainfilename, use_attention, maxseqlen=None, maxclauselen=None, label_ind=None, train=False):
        # list of list
        str_seqs, label_seqs = read_passages(trainfilename, is_labeled=train)
        if not label_ind:
            self.label_ind = {"none": 0}
        else:
            self.label_ind = label_ind
        seq_lengths = [len(seq) for seq in str_seqs]
        if not maxseqlen:
            maxseqlen = max(seq_lengths)
        if not maxclauselen:
            if use_attention:
                clauselens = []
                for str_seq in str_seqs:
                    clauselens.extend([len(clause.split()) for clause in str_seq])
                maxclauselen = max(clauselens)
        X = []
        Y = []
        Y_inds = []
        init_word_rep_len = len(self.rep_reader.word_rep) # Vocab size
        all_word_types = set([])
        for str_seq, label_seq in zip(str_seqs, label_seqs):
            for label in label_seq:
                if label not in self.label_ind:
                    # Add new labels with values 0,1,2,....
                    self.label_ind[label] = len(self.label_ind)
            if use_attention:
                x = np.zeros((maxseqlen, maxclauselen, self.input_size))
            else:
                x = np.zeros((maxseqlen, self.input_size))
            y_ind = np.zeros(maxseqlen)
            seq_len = len(str_seq)
            # The following conditional is true only when we've already trained, and one of the sequences in the test set is longer than the longest sequence in training.
            if seq_len > maxseqlen:
                str_seq = str_seq[:maxseqlen]
                seq_len = maxseqlen
            if train:
                for i, (clause, label) in enumerate(zip(str_seq, label_seq)):
                    clause_rep = self.rep_reader.get_clause_rep(clause) # Makes embedding non-trainable from the beginning.
                    for word in clause.split():
                        all_word_types.add(word) # Vocab
                    if use_attention:
                        if len(clause_rep) > maxclauselen:
                            clause_rep = clause_rep[:maxclauselen]
                        x[-seq_len+i][-len(clause_rep):] = clause_rep
                    else:
                        x[-seq_len+i] = np.mean(clause_rep, axis=0)
                    y_ind[-seq_len+i] = self.label_ind[label]
                X.append(x)
                Y_inds.append(y_ind)
            else:
                for i, clause in enumerate(str_seq):
                    clause_rep = self.rep_reader.get_clause_rep(clause)
                    for word in clause.split():
                        all_word_types.add(word)
                    if use_attention:
                        if len(clause_rep) > maxclauselen:
                            clause_rep = clause_rep[:maxclauselen]
                        x[-seq_len+i][-len(clause_rep):] = clause_rep
                    else:
                        x[-seq_len+i] = np.mean(clause_rep, axis=0)
                X.append(x)
        # Once there is OOV, new word vector is added to word_rep
        final_word_rep_len = len(self.rep_reader.word_rep)
        oov_ratio = float(final_word_rep_len - init_word_rep_len)/len(all_word_types)
        for y_ind in Y_inds:
            y = np.zeros((maxseqlen, len(self.label_ind)))
            for i, y_ind_i in enumerate(y_ind):
                y[i][y_ind_i.astype(int)] = 1
            Y.append(y)
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        return seq_lengths, np.asarray(X), np.asarray(Y) # One-hot representation of labels

    def make_data_cached_elmo(self, use_attention, maxseqlen=300, maxclauselen=30, 
                              label_ind=None, train=False):
        textpath = "/nas/home/xiangcil/bio_corpus/Molecular_Interaction_Evidence_Fragment_Corpus/02_expt_spans_complete/pathway_logic/"
        all_texts = glob.glob(textpath + "*.tsv")
        cachepath = "/nas/home/xiangcil/bio_corpus/Elmo_Cached_Molecular_Interaction_Evidence_Fragment_Corpus/pathway_logic/"
        if not label_ind:
            self.label_ind = {"none": 0}
        else:
            self.label_ind = label_ind
        
        label_seqs = []
        label_seq = []
        elmo_layer = 3
        embedding_dim = 1024
        X = np.zeros((0,maxclauselen,maxseqlen,embedding_dim*elmo_layer))
        for filename in all_texts:
            shortfilename = filename.split("/")[-1].split(".")[0]
            embedding_numpy_file = cachepath + shortfilename + ".npy"
            X_paper = np.load(embedding_numpy_file)
            X = np.append(X,X_paper,axis=0)
            df = pd.read_csv(filename, sep='\t', header=0, index_col=0,engine='python')
            df = df[pd.notnull(df["Discourse Type"])]
            num_rec = df.shape[0]
            prev_paragraph = ""
            for i in range(num_rec):
                if df["Paragraph"][i][0]=="p": # e.g. "p1"
                    if df["Paragraph"][i]!=prev_paragraph:
                        prev_paragraph = df["Paragraph"][i]
                        if len(label_seq)>0:
                            label_seqs.append(label_seq)
                        label_seq = []
                        clause_count = 0
                    if clause_count<maxclauselen:
                        label_seq.append(df["Discourse Type"][i])
                    clause_count += 1
            print("Loading pkl file: ",shortfilename)
        if not use_attention:
            X = np.mean(X,axis=2)
        
        seq_lengths = [len(label_seq) for label_seq in label_seqs]
        Y = []
        Y_inds = []
        for label_seq in label_seqs:
            for label in label_seq:
                if label not in self.label_ind:
                    # Add new labels with values 0,1,2,....
                    self.label_ind[label] = len(self.label_ind)
            y_ind = np.zeros(maxseqlen)
            seq_len = len(label_seq)
            if train:
                for i, label in enumerate(label_seq):
                    y_ind[-seq_len+i] = self.label_ind[label]
                Y_inds.append(y_ind)
        
        for y_ind in Y_inds:
            y = np.zeros((maxseqlen, len(self.label_ind)))
            for i, y_ind_i in enumerate(y_ind):
                y[i][y_ind_i.astype(int)] = 1
            Y.append(y)
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        
        return seq_lengths, X, np.asarray(Y)
    
    def get_attention_weights(self, X_test):
        if not self.tagger:
            raise(RuntimeError, "Tagger not trained yet!")
        inp = self.tagger.get_input()
        att_out = None
        for layer in self.tagger.layers:
            if layer.get_config()['name'].lower() == "tensorattention":
                att_out = layer.get_output()
                break
        if not att_out:
            raise(RuntimeError, "No attention layer found!")
        f = theano.function([inp], att_out)
        return f(X_test)

    def predict(self, X, bidirectional, test_seq_lengths=None, tagger=None):
        if not tagger:
            tagger = self.tagger
        if not tagger:
            raise(RuntimeError, "Tagger not trained yet!")
        if test_seq_lengths is None:
            # Determining actual lengths sans padding
            x_lens = []
            for x in X:
                x_len = 0
                for i, xi in enumerate(x):
                    if xi.sum() != 0:
                        x_len = len(x) - i
                        break
                x_lens.append(x_len)
        else:
                x_lens = test_seq_lengths
        if bidirectional:
            pred_probs = tagger.predict(X)
        else:
            pred_probs = tagger.predict(X)
        pred_inds = np.argmax(pred_probs, axis=2)
        pred_label_seqs = []
        for pred_ind, x_len in zip(pred_inds, x_lens):
            pred_label_seq = [self.rev_label_ind[pred] for pred in pred_ind][-x_len:]
            # If the following number is positive, it means we ignored some clauses in the test passage to make it the same length as the ones we trained on.
            num_ignored_clauses = max(0, x_len - len(pred_label_seq))
            # Make labels for those if needed.
            if num_ignored_clauses > 0:
                warnings.warn("Test sequence too long. Ignoring %d clauses at the beginning and labeling them none." % num_ignored_clauses)
                ignored_clause_labels = ["none"] * num_ignored_clauses
                pred_label_seq = ignored_clause_labels + pred_label_seq
            pred_label_seqs.append(pred_label_seq)
        return pred_probs, pred_label_seqs, x_lens

    def fit_model(self, X, Y, use_attention, att_context, bidirectional, crf):
        early_stopping = EarlyStopping(patience = 2)
        num_classes = len(self.label_ind)
        tagger = Sequential()
        word_proj_dim = 50
        if use_attention:
            sample_size, input_len, timesteps, input_dim = X.shape
            self.td1 = input_len
            self.td2 = timesteps
            tagger.add(HigherOrderTimeDistributedDense(input_dim=input_dim, output_dim=word_proj_dim))
            att_input_shape = (sample_size, input_len, timesteps, word_proj_dim)
            tagger.add(Dropout(0.5))
            tagger.add(TensorAttention(att_input_shape, context=att_context))
        else:
            _, input_len, input_dim = X.shape
            tagger.add(TimeDistributed(Dense(input_dim=input_dim, units=word_proj_dim)))
        if bidirectional:
            tagger.add(Bidirectional(LSTM(input_shape=(input_len,word_proj_dim), units=word_proj_dim, return_sequences=True)))
        else:
            tagger.add(LSTM(input_shape=(input_len,word_proj_dim), units=word_proj_dim, return_sequences=True))
        tagger.add(TimeDistributed(Dense(num_classes, activation='softmax')))
        def step_decay(epoch):
            initial_lrate = 0.1
            drop = 0.5
            epochs_drop = 5.0
            lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
            return lrate
        epoch = 100
        if crf:
            crf = CRF(num_classes, learn_mode="marginal")
            tagger.add(crf)
            #rmsprop = RMSprop(lr=0.05, rho=0.9, epsilon=None, decay=0.99)
            #lr = 0.1
            #decay = lr / epoch
            #sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
            tagger.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
            
        else:
            tagger.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #tagger.fit(X, Y, validation_split=0.1, epochs=100, callbacks=[early_stopping], batch_size=10)

        #tagger.fit(X, Y, validation_split=0.1, epochs=epoch, batch_size=10, callbacks = [LearningRateScheduler(step_decay)])
        tagger.fit(X, Y, validation_split=0.1, epochs=epoch, batch_size=10)
        tagger.summary()
        return tagger

    def train(self, X, Y, use_attention, att_context, bidirectional, cv=True, folds=5, crf=False):
        if cv:
            cv_folds = make_folds(X, Y, folds)
            accuracies = []
            fscores = []
            for fold_num, ((train_fold_X, train_fold_Y), (test_fold_X, test_fold_Y)) in enumerate(cv_folds):
                self.tagger = self.fit_model(train_fold_X, train_fold_Y, use_attention, att_context, bidirectional, crf)
                pred_probs, pred_label_seqs, x_lens = self.predict(test_fold_X, bidirectional, tagger=self.tagger)
                pred_inds = np.argmax(pred_probs, axis=2)
                flattened_preds = []
                flattened_targets = []
                for x_len, pred_ind, test_target in zip(x_lens, pred_inds, test_fold_Y):
                    flattened_preds.extend(pred_ind[-x_len:])
                    flattened_targets.extend([list(tt).index(1) for tt in test_target[-x_len:]])
                assert len(flattened_preds) == len(flattened_targets)
                accuracy, weighted_fscore, all_fscores = evaluate(flattened_targets, flattened_preds)
                print("Finished fold %d. Accuracy: %f, Weighted F-score: %f"%(fold_num, accuracy, weighted_fscore))
                print("Individual f-scores:")
                for cat in all_fscores:
                    print("%s: %f"%(self.rev_label_ind[cat], all_fscores[cat]))
                accuracies.append(accuracy)
                fscores.append(weighted_fscore)
            accuracies = np.asarray(accuracies)
            fscores = np.asarray(fscores)
            print("Accuracies:", accuracies)
            print("Average: %0.4f (+/- %0.4f)"%(accuracies.mean(), accuracies.std() * 2))
            print(sys.stderr, "Fscores:", fscores)
            print(sys.stderr, "Average: %0.4f (+/- %0.4f)"%(fscores.mean(), fscores.std() * 2))
        else:
            self.tagger = self.fit_model(X, Y, use_attention, att_context, bidirectional,crf)
        model_ext = "att=%s_cont=%s_bi=%s"%(str(use_attention), att_context, str(bidirectional))
        model_config_file = open("model_%s_config.json"%model_ext, "w")
        model_weights_file_name = "model_%s_weights"%model_ext
        model_label_ind = "model_%s_label_ind.json"%model_ext
        model_rep_reader = "model_%s_rep_reader.pkl"%model_ext
        self.tagger.save_weights(model_weights_file_name, overwrite=True)
        json.dump(self.label_ind, open(model_label_ind, "w"))
        pickle.dump(self.rep_reader, open(model_rep_reader, "wb"))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run LSTM discourse tagger")
    argparser.add_argument('--repfile', type=str, help="Gzipped word embedding file")
    argparser.add_argument('--train_file', type=str, help="Training file. One clause<tab>label per line and passages separated by blank lines.")
    argparser.add_argument('--cv', help="Do cross validation", action='store_true')
    argparser.add_argument('--test_files', metavar="TESTFILE", type=str, nargs='+', help="Test file name(s), separated by space. One clause per line and passages separated by blank lines.")
    argparser.add_argument('--use_attention', help="Use attention over words? Or else will average their representations", action='store_true')
    argparser.add_argument('--att_context', type=str, help="Context to look at for determining attention (word/clause)")
    argparser.set_defaults(att_context='word')
    argparser.add_argument('--bidirectional', help="Bidirectional LSTM", action='store_true')
    argparser.add_argument('--show_attention', help="When testing, if using attention, also print the weights", action='store_true')
    argparser.add_argument('--crf', help="Conditional Random Field", action='store_true')
    args = argparser.parse_args()
    repfile = args.repfile
    if args.train_file:
        trainfile = args.train_file
        train = True
        #assert args.repfile is not None, "Word embedding file required for training."
    else:
        train = False
    if args.test_files:
        testfiles = args.test_files
        test = True
    else:
        test = False
    if not train and not test:
        raise(RuntimeError, "Please specify a train file or test files.")
    use_attention = args.use_attention
    att_context = args.att_context
    bid = args.bidirectional
    show_att = args.show_attention
    crf = args.crf

    if train:
        # First returned value is sequence lengths (without padding)
        nnt = PassageTagger(word_rep_file=repfile)
        if repfile:
            print("Using embedding weight to find embeddings.")
            _, X, Y = nnt.make_data(trainfile, use_attention, train=True)
        else:
            print("Load cached Elmo embedding.")
            _, X, Y = nnt.make_data_cached_elmo(use_attention, train=True)
        nnt.train(X, Y, use_attention, att_context, bid, cv=args.cv, crf=crf)
    if test:
        if train:
            label_ind = nnt.label_ind
            print("label_ind",label_ind)
        else:
            # Load the model from file
            model_ext = "att=%s_cont=%s_bi=%s"%(str(use_attention), att_context, str(bid))
            model_config_file = open("model_%s_config.json"%model_ext, "r")
            model_weights_file_name = "model_%s_weights"%model_ext
            model_label_ind = "model_%s_label_ind.json"%model_ext
            model_rep_reader = "model_%s_rep_reader.pkl"%model_ext
            rep_reader = pickle.load(open(model_rep_reader, "rb"))
            print("Loaded pickled rep reader")
            nnt = PassageTagger(pickled_rep_reader=rep_reader)
            nnt.tagger = model_from_json(model_config_file.read(), custom_objects={"TensorAttention":TensorAttention, "HigherOrderTimeDistributedDense":HigherOrderTimeDistributedDense})
            print("Loaded model:")
            print(nnt.tagger.summary())
            nnt.tagger.load_weights(model_weights_file_name)
            print("Loaded weights")
            label_ind_json = json.load(open(model_label_ind))
            label_ind = {k: int(label_ind_json[k]) for k in label_ind_json}
            print("Loaded label index:", label_ind)
        if not use_attention:
            assert nnt.tagger.layers[0].name == "timedistributeddense"
            maxseqlen = nnt.tagger.layers[0].input_length
            maxclauselen = None
        else:
            maxseqlen, maxclauselen = nnt.td1, nnt.td2
        for test_file in testfiles:
            print("Predicting on file %s"%(test_file))
            test_out_file_name = test_file.split("/")[-1].replace(".txt", "")+"_att=%s_cont=%s_bid=%s"%(str(use_attention), att_context, str(bid))+".out"
            outfile = open(test_out_file_name, "w")
            test_seq_lengths, X_test, _ = nnt.make_data(test_file, use_attention, maxseqlen=maxseqlen, maxclauselen=maxclauselen, label_ind=label_ind, train=False)
            print("X_test shape:", X_test.shape)
            pred_probs, pred_label_seqs, _ = nnt.predict(X_test, bid, test_seq_lengths)
            if show_att:
                att_weights = nnt.get_attention_weights(X_test.astype('float32'))
                clause_seqs, _ = read_passages(test_file, is_labeled=True)
                paralens = [[len(clause.split()) for clause in seq] for seq in clause_seqs]
                for clauselens, sample_att_weights, pred_label_seq in zip(paralens, att_weights, pred_label_seqs):
                    for clauselen, clause_weights, pred_label in zip(clauselens, sample_att_weights[-len(clauselens):], pred_label_seq):
                        print(outfile, pred_label, " ".join(["%.4f"%val for val in clause_weights[-clauselen:]]))
                    print(outfile)
            else:
                for pred_label_seq in pred_label_seqs:
                    for pred_label in pred_label_seq:
                        print(pred_label,file=outfile)
