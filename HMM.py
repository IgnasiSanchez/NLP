import numpy as np

def logzero():
    return -np.inf


def safe_log(x):
    print(x)
    if x == 0:
        return logzero()
    return np.log(x)


def logsum_pair(logx, logy):
    """
    Return log(x+y), avoiding arithmetic underflow/overflow.

    logx: log(x)
    logy: log(y)

    Rationale:

    x + y    = e^logx + e^logy
             = e^logx (1 + e^(logy-logx))
    log(x+y) = logx + log(1 + e^(logy-logx)) (1)

    Likewise,
    log(x+y) = logy + log(1 + e^(logx-logy)) (2)

    The computation of the exponential overflows earlier and is less precise
    for big values than for small values. Due to the presence of logy-logx
    (resp. logx-logy), (1) is preferred when logx > logy and (2) is preferred
    otherwise.
    """
    if logx == logzero():
        return logy
    elif logx > logy:
        return logx + np.log1p(np.exp(logy-logx))
    else:
        return logy + np.log1p(np.exp(logx-logy))


def logsum(logv):
    """
    Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.
    """
    res = logzero()
    for val in logv:
        res = logsum_pair(res, val)
    return res

class HMM(object):
    
    def __init__(self, word_to_pos={}, state_to_pos={}):
        self.fitted = False
        self.counts = {"emission": None, "transition":None, "final":None, "initial":None}
        self.probs  = {"emission": None, "transition":None, "final":None, "initial":None}
        self.scores = {"emission": None, "transition":None, "final":None, "initial":None}
        self.decode = set(["posterior", "viterbi"])
        self.word_to_pos  = word_to_pos
        self.state_to_pos = state_to_pos
        self.pos_to_word  = {v: k for k, v in word_to_pos.items()}
        self.pos_to_state = {v: k for k, v in state_to_pos.items()}
    
        self.n_states     = len(state_to_pos)
        self.n_words      = len(word_to_pos)
        self.fitted = False

    def fit(self, observation_lables: list, state_labels: list):
        """
        Computes and saves: counts, probs, scores.
        """
        if self.state_to_pos is None or self.word_to_pos is None:
            print("Error state_to_pos or word_to_pos needed to be defined")
            return
            
        self.counts = self.sufficient_statistics_hmm(observation_lables, state_labels)       
        self.probs  = self.compute_probs(self.counts)  
        self.scores = self.compute_scores(self.probs)  
        self.fitted = True
        
    def sufficient_statistics_hmm(self, observation_lables, state_labels):

        state_to_pos, word_to_pos = self.state_to_pos, self.word_to_pos
        
        def update_initial_counts(initial_counts, seq_y, state_to_pos):    
            initial_counts[state_to_pos[seq_y[0]]] += 1

        def update_transition_counts(transition_counts, seq_y, state_to_pos):
            for (t_prev, t) in zip(seq_y[:-1], seq_y[1:]):
                transition_counts[state_to_pos[t], state_to_pos[t_prev]] += 1

        def update_emission_counts(emission_counts, seq_x, seq_y, state_to_pos, word_to_pos):
            for (t,w) in zip(seq_y, seq_x):
                emission_counts[state_to_pos[t], word_to_pos[w]] += 1

        def update_final_counts(final_counts, seq_y, state_to_pos):
            final_counts[state_to_pos[seq_y[-1]]] += 1


        n_states = len(state_to_pos)
        n_words  = len(word_to_pos)
        initial_counts      = np.zeros((n_states))
        transition_counts   = np.zeros((n_states, n_states))
        final_counts        = np.zeros((n_states))
        emission_counts     = np.zeros((n_states, n_words))

        for seq_x, seq_y in zip(observation_lables, state_labels):
            update_initial_counts(initial_counts, seq_y, state_to_pos)
            update_transition_counts(transition_counts, seq_y,  state_to_pos)
            update_emission_counts(emission_counts, seq_x, seq_y, state_to_pos, word_to_pos) 
            update_final_counts(final_counts, seq_y,  state_to_pos) 

        return {"emission":   emission_counts, 
                "transition": transition_counts,
                "final":      final_counts, 
                "initial":    initial_counts}
    
    def compute_probs(self, counts):
        
        initial_counts    = counts['initial']
        transition_counts = counts['transition']
        emission_counts   = counts['emission']
        final_counts      = counts['final']

        initial_probs    = (initial_counts / np.sum(initial_counts))
        transition_probs = transition_counts/(np.sum(transition_counts,0) + final_counts)
        final_probs      = final_counts/(np.sum(transition_counts, 0) + final_counts )
        emission_probs   = (emission_counts.T / np.sum(emission_counts, 1)).T
    
        return {"emission":   emission_probs, 
                "transition": transition_probs,
                "final":      final_probs, 
                "initial":    initial_probs}
    
    def compute_scores(self, probs):
         return {"emission":   np.log(probs["emission"]), 
                 "transition": np.log(probs["transition"]),
                 "final":      np.log(probs["final"]), 
                 "initial":    np.log(probs["initial"])}
        
    def forward_computations(self, x: list):
        forward_x = None
        return forward_x
    
    def backward_computations(self, x:list):
        backward_x = None
        return backward_x
    
    def log_forward_computations(self, x: list):
        """
        Compute the log_forward computations

        Assume there are S possible states and a sequence of length N.
        This method will compute iteritavely the log_forward quantities.

        * log_f is a S x N Array.
        * log_f_x[:,i] will contain the forward quantities at position i.
        * log_f_x[:,i] is a vector of size S.
        
        Returns
        - log_f_x: Array of size K x N
        """ 
        n_x = len(x)
        
        # log_f_x initialized to -Inf because log(0) = -Inf
        log_f_x = np.zeros((self.n_states, n_x)) - np.Inf
        x_emission_scores = np.array([self.scores['emission'][:, self.word_to_pos[w]] for w in x]).T
        
        log_f_x[:,0] = x_emission_scores[:, 0] + self.scores['initial']
        for n in range(1, n_x):
            for s in range(self.n_states):
                log_f_x[s,n] = logsum(log_f_x[:,n-1] + self.scores['transition'][s,:] + x_emission_scores[s,n])
        
        log_likelihood = logsum(log_f_x[:,n_x-1] + self.scores['final'])

        return log_f_x, log_likelihood
    
    
    def log_backward_computations(self, x: list):
        n_x = len(x)
        
        # log_f_x initialized to -Inf because log(0) = -Inf
        log_b_x = np.zeros((self.n_states, n_x)) - np.Inf
        x_emission_scores = np.array([self.scores['emission'][:, self.word_to_pos[w]] for w in x]).T
        log_b_x[:,-1] = self.scores['final']
        for n in range(n_x-2, 0, -1):
            for s in range(self.n_states):
                log_b_x[s,n] = logsum(log_b_x[:,n+1] + self.scores['transition'][:,s] + x_emission_scores[:,n+1])
        
        log_likelihood = logsum(log_b_x[:,1] + self.scores['initial'])
        
        return log_b_x, log_likelihood
        
    def predict_labels(self, x: list, decode="posterior"):
        """
        Retuns a sequence of states for each word in **x**.
        The output depends on the **decode** method chosen.
        """
        assert decode in self.decode, "decode `{}` is not valid".format(decode)
        
        if decode is 'posterior':
            return self.posterior_decode(x)
        
        if decode is 'viterbi':
            return self.viterbi_decode(x)

    def compute_state_posteriors(self, x:list):
        log_f_x, log_likelihood = self.log_forward_computations(x)
        log_b_x, log_likelihood = self.log_backward_computations(x)
        state_posteriors = np.zeros((self.n_states, len(x)))
        
        for pos in range(len(x)):
            state_posteriors[:, pos] = log_f_x[:, pos] + log_b_x[:, pos] - log_likelihood
        return state_posteriors

    def posterior_decode(self, x: list, decode_states=True):
        
        state_posteriors = self.compute_state_posteriors(x)
        y_hat = state_posteriors.argmax(axis=0)
        
        if decode_states:
            y_hat = [self.pos_to_state[y] for y in y_hat]
            
        return y_hat