import time
import numpy as np
import scipy.special
import re
from scipy.special import gamma, gammaln


class ASUMCaseGibbs:
    """
    Explainable Aspect and Sentiment Unification Model for Online Review Analysis
    """
    def __init__(self, num_topics, senti_words, doc_file_path, vocas,
                 alpha=0.1, beta=0.01, gamma=0.1, high_beta=0.7, q=0.8, Vj=5, c=10.0, seed=108):
        """
        Constructor method
        :param num_topics: the number of topics
        :param doc_file_path: BOW document file path
        :param vocas: vocabulary list
        :param alpha: alpha value in ASUM
        :param beta: beta value in ASUM
        :param gamma: gamma value in ASUM
        :param high_beta: beta value in ASUM for sentiment words
        :param q: q value in ASUM
        :param Vj: Vj value in ASUM
        :param c: c value in ASUM
        :param seed: random seed value
        :return: void
        """
        np.random.seed(seed)

        self.docs = self.read_bow(doc_file_path)
        self.words = vocas
        self.K = num_topics
        self.D = len(self.docs)
        self.W = len(vocas)
        self.S = len(senti_words)
        self.q = q
        self.Vj = Vj
        self.c = c
        word_idx = 0
        self.words_to_idx = dict()
        for word in self.words:
            self.words_to_idx[word] = word_idx
            word_idx += 1

        # Hyper-parameters
        self.alpha = alpha
        self.beta = np.zeros((self.S, self.W)) + beta
        for senti_idx, one_senti_words in enumerate(senti_words):
            for one_senti_word in one_senti_words:
                try:
                    self.beta[senti_idx, self.words_to_idx[one_senti_word]] = high_beta
                except KeyError:
                    pass
        self.gamma = gamma
        self.g_mat_origin = np.repeat(self.beta[:, np.newaxis, :], self.K, axis=1)

        self.DSK = np.zeros((self.D, self.S, self.K), dtype=np.int64)
        self.SKW = np.zeros((self.S, self.K, self.W), dtype=np.int64)
        self.prototype_docs = np.array(range(self.S * self.K))
        self.omega = np.random.binomial(1, self.q, size=(self.S, self.K, self.W))

        # Random initialization of topics
        self.doc_topics = list()

        for di in range(self.D):
            doc = self.docs[di]

            topics = np.random.randint(self.S * self.K, size=len(doc))
            self.doc_topics.append(topics)

            for senti_topic, sentence in zip(topics, doc):
                senti = senti_topic // self.K
                topic = senti_topic % self.K
                self.DSK[di, senti, topic] += 1

                target_mat = self.SKW[senti, topic, :]
                for word_idx, word_cnt in sentence:
                    target_mat[word_idx] += word_cnt

    def run(self, max_iter=2000, do_optimize=False, do_print_log=False):
        """
        Run Collapsed Gibbs sampling for ASUM
        :param max_iter: Maximum number of gibbs sampling iteration
        :param do_optimize: Do run optimize hyper-parameters
        :param do_print_log: Do print loglikelihood and run time
        :return: void
        """
        if do_optimize and do_print_log:
            prev = time.clock()
            for iteration in range(max_iter):
                print(iteration, time.clock() - prev, self.loglikelihood())
                prev = time.clock()
                self._gibbs_sampling()
                if 99 == iteration % 100:
                    self._optimize()
        elif do_optimize and not do_print_log:
            for iteration in range(max_iter):
                self._gibbs_sampling()
                if 99 == iteration % 100:
                    self._optimize()
        elif not do_optimize and do_print_log:
            prev = time.clock()
            for iteration in range(max_iter):
                print(iteration, time.clock() - prev, self.loglikelihood())
                prev = time.clock()
                self._gibbs_sampling()
        else:
            prev = time.clock()
            for iteration in range(max_iter):
                print(iteration, time.clock() - prev)
                prev = time.clock()
                self._gibbs_sampling()

    def _gibbs_sampling(self):
        # For z and s
        prev_time = time.time()
        self._sampling_z_s()
        print("Sampling time for z: {}".format(time.time() - prev_time))

        # For omega
        prev_time = time.time()
        self._sampling_omega()
        print("Sampling time for omega: {}".format(time.time() - prev_time))

        # For p
        prev_time = time.time()
        self._sampling_p()
        print("Sampling time for p: {}".format(time.time() - prev_time))

    def _sampling_z_s(self):
        """
        Run Gibbs Sampling for z and s
        :return: void
        """
        for di in range(self.D):
            doc = self.docs[di]
            cur_doc_senti_topics = self.doc_topics[di]

            for sentence_idx, sentence in enumerate(doc):
                # Old one
                old_senti_topic = cur_doc_senti_topics[sentence_idx]
                senti = old_senti_topic // self.K
                topic = old_senti_topic % self.K

                self.DSK[di, senti, topic] -= 1
                target_mat = self.SKW[senti, topic, :]
                for word_idx, word_cnt in sentence:
                    target_mat[word_idx] -= word_cnt

                g_func_mat = self._compute_g_func()

                first_term = np.sum(self.DSK[di, :, :], axis=1, keepdims=True) + self.gamma
                second_term_part = self.DSK[di, :, :] + self.alpha
                second_term = second_term_part / (np.sum(second_term_part, axis=1, keepdims=True))
                third_term_part = np.sum(self.SKW, axis=2) + np.sum(g_func_mat, axis=2, keepdims=True)
                forth_term = 1
                words_in_doc = 0
                for word_idx, word_cnt in sentence:
                    words_in_doc += word_cnt
                    forth_term_part = self.SKW[:, :, word_idx] + g_func_mat[:, word_idx]
                    temp_prod = 1
                    for cnt_idx in range(word_cnt):
                        temp_prod *= (forth_term_part + cnt_idx)
                    forth_term *= temp_prod
                third_term = scipy.special.gamma(third_term_part) / (scipy.special.gamma(third_term_part + words_in_doc))

                # Sampling
                prob = first_term * second_term * third_term * forth_term
                prob = prob.flatten()

                # New one
                new_senti_topic = self._sampling_from_dist(prob)
                senti = new_senti_topic // self.K
                topic = new_senti_topic % self.K

                cur_doc_senti_topics[sentence_idx] = new_senti_topic

                self.DSK[di, senti, topic] += 1
                target_mat = self.SKW[senti, topic, :]
                for word_idx, word_cnt in sentence:
                    target_mat[word_idx] += word_cnt

    @staticmethod
    def _sampling_from_dist(prob):
        """
        Multinomial sampling with probability vector
        :param prob: probability vector
        :return: a new sample (In this class, it is new topic index)
        """
        thr = prob.sum() * np.random.rand()
        new_topic = 0
        tmp = prob[new_topic]
        while tmp < thr:
            new_topic += 1
            tmp += prob[new_topic]
        return new_topic

    def _compute_g_func(self):
        g_mat = self.g_mat_origin.copy()
        omega_vec = self.omega

        for senti_topic_idx, proto_idx in enumerate(self.prototype_docs):
            senti_idx = senti_topic_idx // self.K
            topic_idx = senti_topic_idx % self.K
            for word_idx in self.W:
                if 1 == omega_vec[senti_idx, topic_idx, word_idx]:
                    proto_doc = self.docs[proto_idx]
                    for sentence in proto_doc:
                        if word_idx in sentence:
                            g_mat[senti_idx, topic_idx, word_idx] += self.c
                            break
        return g_mat

    def _compute_one_g_func_with_omega(self, senti_idx, topic_idx, word_idx, omega):
        """
        Compute g function
        :param senti_idx: sentiment index
        :param topic_idx: topic index
        :param word_idx: word index
        :param omega: omega for the topic
        :return: computed vector as parameter of dirichlet distribution
        """
        return_vec = np.zeros(2) + self.g_mat_origin[senti_idx, topic_idx, word_idx]

        if 1 == omega:
            proto_doc = self.docs[self.prototype_docs[senti_idx * topic_idx]]
            for sentence in proto_doc:
                if word_idx in sentence:
                    return_vec[1] += self.c
                    break

        return return_vec

    def _sampling_omega(self):
        """
        Sampling omega
        :return:
        """
        prob = np.zeros(2)

        for senti_idx in range(self.S):
            for topic_idx in range(self.K):
                for word_idx in range(self.W):
                    vec_g = self._compute_one_g_func_with_omega(senti_idx, topic_idx, word_idx, 0)
                    vec_g_n = vec_g + self.SKW[senti_idx, topic_idx, word_idx]
                    first_term = gamma(vec_g.sum()) / np.prod(gamma(vec_g))
                    second_term = np.prod(gamma(vec_g_n)) / gamma(vec_g_n.sum())
                    prob[0] = (1 - self.q) * first_term * second_term

                    vec_g = self._compute_one_g_func_with_omega(topic_idx, word_idx, 1)
                    vec_g_n = vec_g + self.SKW[senti_idx, topic_idx, word_idx]
                    first_term = gamma(vec_g.sum()) / np.prod(gamma(vec_g))
                    second_term = np.prod(gamma(vec_g_n)) / gamma(vec_g_n.sum())
                    prob[1] = self.q * first_term * second_term

                    prob = prob / prob.sum()

                    self.omega[senti_idx, topic_idx, word_idx] = np.random.multinomial(1, prob).argmax()

    def _sampling_p(self):
        """
        Log prob
        Sampling prototype p
        :return:
        """
        for senti_idx in range(self.S):
            for topic_idx in range(self.K):
                prob = np.zeros(self.D)
                for doc_idx, doc in enumerate(self.docs):
                    for word_idx, omega_val in enumerate(self.omega[senti_idx, topic_idx, :]):
                        if 0 == omega_val:
                            continue

                        vec_g = np.zeros(2) + self.g_mat_origin[senti_idx, topic_idx, word_idx]
                        for sentence in doc:
                            if word_idx in sentence:
                                vec_g[1] += self.c
                                break
                        vec_g_n = vec_g + self.SKW[senti_idx, topic_idx, word_idx]
                        first_term = gammaln(vec_g.sum()) - np.sum(gammaln(vec_g))
                        second_term = np.sum(gammaln(vec_g_n)) - gammaln(vec_g_n.sum())
                        prob[doc_idx] += first_term + second_term

                prob = prob / prob.sum()

                self.prototype_docs[topic_idx] = np.random.multinomial(1, prob).argmax()

    def loglikelihood(self):
        """
        Compute log likelihood function
        :return: log likelihood function
        """
        return self._topic_loglikelihood() + self._document_loglikelihood()

    def _topic_loglikelihood(self):
        """
        Compute log likelihood by topics
        :return: log likelihood by topics
        """
        raise NotImplementedError

    def _document_loglikelihood(self):
        """
        Compute log likelihood by documents
        :return: log likelihood by documents
        """
        raise NotImplementedError

    def _optimize(self):
        """
        Optimize hyperparameters
        :return: void
        """
        self._alphaoptimize()
        self._betaoptimize()
        self._gammaoptimize()

    def _alphaoptimize(self, conv_threshold=0.001):
        """
        Optimize alpha vector
        :return: void
        """
        raise NotImplementedError

    def _betaoptimize(self, conv_threshold=0.001):
        """
        Optimize beta value
        :return: void
        """
        raise NotImplementedError

    def _gammaoptimize(self, conv_threshold=0.001):
        """
        Optimize gamma value
        :return: void
        """
        raise NotImplementedError

    def export_result(self, output_file_name, rank_idx=100):
        """
        Export Algorithm Result to File
        :param output_file_name: output file name
        :param rank_idx:
        :return: the number of printed words in a topic in output file
        """
        # Raw data
        np.save("{}_DSK.npy".format(output_file_name), self.DSK)
        np.save("{}_SKW.npy".format(output_file_name), self.SKW)
        np.save("{}_omega.npy".format(output_file_name), self.omega)
        np.savetxt("{}_prototypes.csv".format(output_file_name), self.prototype_docs, delimiter=",")

        # Ranked words in topics
        with open("%s_Topic_Ranked.csv" % output_file_name, "w") as ranked_topic_word_file:
            for senti_idx in range(self.S):
                for topic_idx in range(self.K):
                    topic_vec = self.SKW[senti_idx, topic_idx, :]
                    sorted_words = sorted(enumerate(topic_vec), key=lambda x: x[1], reverse=True)
                    print('senti/topic {}/{},{}'.format(senti_idx, topic_idx,
                                                        ",".join([self.words[x[0]] for x in sorted_words[:rank_idx]])),
                          file=ranked_topic_word_file)

    @staticmethod
    def read_bow(file_path):
        """
        Read BOW file to run topic models with Gibbs sampling
        :param file_path: The path of BOW file
        :return: documents list
        """
        split_pattern = re.compile(r'[ :]')
        docs = list()

        with open(file_path, 'r') as bow_file:
            for each_line in bow_file:
                one_doc = list()
                sentences = each_line.split(",")[2:]
                for each_sentence in sentences:
                    split_line = split_pattern.split(each_sentence)
                    word_ids = [int(x) for x in split_line[0::2]]
                    word_counts = [int(x) for x in split_line[1::2]]

                    one_doc.append(zip(word_ids, word_counts))

                docs.append(one_doc)

        return docs
