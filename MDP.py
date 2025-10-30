import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        V = initialV.copy()
        iterId = 0
        epsilon = np.inf

        while(iterId < nIterations and epsilon > tolerance):
            Ta_V = np.matmul(self.T, V)     # T和V矩陣相乘可一次性計算出了所有狀態下，採取所有動作的未來期望價值。
            All_values = self.R + (self.discount * Ta_V)   # 乘上gamma再加上reward可得|A| x |S|價值矩陣
            V_new = np.max((All_values), axis=0)       # 找出每個S的最大action value
            epsilon = np.max(np.abs(V_new - V))
            V = V_new
            iterId = iterId + 1

        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''
        
        policy = np.zeros(self.nStates)

        # 計算|A| x |S|價值矩陣
        Ta_V = np.matmul(self.T, V)     
        All_values = self.R + (self.discount * Ta_V) 
        # 找出最大action value的位置
        policy = np.argmax(All_values, axis=0)
        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''
        # 利用移項直接解V^pi = (I - gamma T^pi)^-1 R^pi
        # 先計算出policy的reward和狀態轉移矩陣
        R_policy = np.zeros(self.nStates)
        T_policy = np.zeros((self.nStates, self.nStates))
        for i in range(len(policy)):
            action = policy[i]
            R_policy[i] = self.R[action, i]
            T_policy[i] = self.T[action, i]
        gamma_T_policy = self.discount * T_policy
        # 算反矩陣(I - gamma T^pi)^-1
        invert = np.linalg.inv(np.identity(self.nStates) - gamma_T_policy)

        V = np.matmul(invert, R_policy)

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        policy = initialPolicy
        V = np.zeros(self.nStates)
        iterId = 0

        while(iterId < nIterations):
            V = self.evaluatePolicy(policy)
            new_policy = self.extractPolicy(V)
            iterId = iterId + 1
            if(np.array_equal(new_policy, policy)):
                break
            policy = new_policy.copy()

        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        
        V = initialV
        iterId = 0
        epsilon = np.inf
        # 先計算出policy的reward和狀態轉移矩陣
        R_policy = np.zeros(self.nStates)
        T_policy = np.zeros((self.nStates, self.nStates))
        for i in range(len(policy)):
            action = policy[i]
            R_policy[i] = self.R[action, i]
            T_policy[i] = self.T[action, i]
        # 計算value直到收斂
        while(iterId < nIterations and epsilon > tolerance):
            iterId += 1
            V_new = R_policy + self.discount * np.matmul(T_policy, V)
            epsilon = np.max(np.abs(V_new - V))
            V = V_new


        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        policy = initialPolicy
        V = initialV
        iterId = 0
        epsilon = np.inf

        while(iterId < nIterations and epsilon > tolerance):
            iterId = iterId + 1
            V_new, _, _ = self.evaluatePolicyPartially(policy, V, nEvalIterations, tolerance)
            # extract Policy
            Ta_V = np.matmul(self.T, V_new)     
            All_values = self.R + (self.discount * Ta_V) 
            policy = np.argmax(All_values, axis=0)
            # use new policy to get value
            V_new_policy = [All_values[policy[i]][i] for i in range(len(policy))]
            epsilon = np.max(np.abs(V_new_policy - V_new))
            V = V_new_policy

        return [policy,V,iterId,epsilon]