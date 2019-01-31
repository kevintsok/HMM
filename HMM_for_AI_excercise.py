import numpy as np

s = ['L', 'B', 'k']
o = ['L', 'B', 'k']





'''
start = {'L': 0.0, 'B': 0.0, 'k': 1.0}
trans = {
    'L': {'L': 0.1, 'B': 0.3, 'k': 0.6},
    'B': {'L': 0.3, 'B': 0.1, 'k': 0.6},
    'k': {'L': 0.1, 'B': 0.1, 'k': 0.8}
}
emiss = {
    'L': {'L': 0.6, 'B': 0.3, 'k': 0.1},
    'B': {'L': 0.4, 'B': 0.5, 'k': 0.1},
    'k': {'L': 0.1, 'B': 0.1, 'k': 0.8}
}
'''

start = np.array([0.0, 0.0, 1.0])
trans = [
    [0.1, 0.3, 0.6],
    [0.3, 0.1, 0.6],
    [0.1, 0.1, 0.8]
]
trans = np.array(trans)
emiss = [
    [0.6, 0.3, 0.1],
    [0.4, 0.5, 0.1],
    [0.1, 0.1, 0.8]
]
emiss = np.array(emiss)


def convert_num_to_string(vec,string):
    return [string[i] for i in vec]

def predict_tstate(T):
    N = len(s)
    st = np.zeros((N,T))
    st[:, 0] = start
    for t in range(1,T):
        st[:, t] = np.dot(st[:, t-1], trans)
    return st

def forward(obs):
    N = len(s)
    T = len(obs)
    F = np.zeros((N,T))
    F[:, 0] = start*emiss[:, obs[0]]
    for t in range(1, T):
        for n in range(N):
            F[n,t] = np.dot(F[:,t-1],trans[:,n])*emiss[n, obs[t]]
    return F

def backward(obs):
    N = len(s)
    T = len(obs)
    B = np.zeros((N,T))
    B[:,-1:]=1
    for t in range(T-1)[::-1]:
        for n in range(N):
            B[n,t]=np.sum(B[:,t+1]*trans[n,:]*emiss[:, obs[t+1]])
    return B


print(predict_tstate(3))

obs = [2,1,1]
print(convert_num_to_string(obs, s))
F=forward(obs)
print('\nForward:')
print(F)

res_forward = 0
for i in range(3):                         #将最后一列相加就得到了我们最终的结果
    res_forward+=F[i][len(obs)-1]
print(res_forward)
#print(np.divide(F,sum(F,0)))

print('\nBackward:')
B=backward(obs)
print(B)
print(np.sum(start*B[:,0]*emiss[:,obs[0]]))


def viterbi(obs_seq, A, B, pi):
    """
    Returns
    -------
    V : numpy.ndarray
        V [s][t] = Maximum probability of an observation sequence ending
                   at time 't' with final state 's'
    prev : numpy.ndarray
        Contains a pointer to the previous state at t-1 that maximizes
        V[state][t]

    V对应δ，prev对应ψ
    """
    N = A.shape[0]
    T = len(obs_seq)
    prev = np.zeros((T - 1, N), dtype=int)

    # DP matrix containing max likelihood of state at a given time
    V = np.zeros((N, T))
    V[:, 0] = pi * B[:, obs_seq[0]]

    for t in range(1, T):
        for n in range(N):
            seq_probs = V[:, t - 1] * A[:, n] * B[n, obs_seq[t]]
            prev[t - 1, n] = np.argmax(seq_probs)
            V[n, t] = np.max(seq_probs)

    return V, prev


def build_viterbi_path(prev, last_state):
    """Returns a state path ending in last_state in reverse order.
    最优路径回溯
    """
    T = len(prev)
    yield (last_state)
    for i in range(T - 1, -1, -1):
        yield (prev[i, last_state])
        last_state = prev[i, last_state]


def observation_prob(obs_seq):
    """ P( entire observation sequence | A, B, pi ) """
    return np.sum(forward(obs_seq)[:, -1])


def state_path(obs_seq, A, B, pi):
    """
    Returns
    -------
    V[last_state, -1] : float
        Probability of the optimal state path
    path : list(int)
        Optimal state path for the observation sequence
    """
    V, prev = viterbi(obs_seq, A, B, pi)
    # Build state path with greatest probability
    last_state = np.argmax(V[:, -1])
    path = list(build_viterbi_path(prev, last_state))

    return V[last_state, -1], reversed(path)

V, p = viterbi(obs, trans, emiss, start)
p,ss = state_path(obs, trans, emiss, start)

print('\nViterbi:')
print(V)

for s in ss:
    print(s)
print(p)
#print(p / len(states_data))
