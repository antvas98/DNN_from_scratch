import numpy as np

class DNN:
    def __init__(self, input_dim=2, layers=2, width=4):
        self.L = layers
        self.width = width
        self.input_dim = input_dim
        self.W_init = []
        self.b_init = []
        self.initialize_weights()

    def initialize_weights(self):
        self.W_init.append(np.random.uniform(-1/np.sqrt(self.width)-1, 1/np.sqrt(self.width)+1, (self.width, self.input_dim)))
        self.b_init.append(np.random.uniform(-1/np.sqrt(self.width)-1, 1/np.sqrt(self.width)+1, (self.width, 1)))

        for i in range(self.L-1):
            self.W_init.append(np.random.uniform(-1/np.sqrt(self.width)-1, 1/np.sqrt(self.width)+1, (self.width, self.width)))
            self.b_init.append(np.random.uniform(-1/np.sqrt(self.width)-1, 1/np.sqrt(self.width)+1, (self.width, 1)))

        self.W_init.append(np.random.uniform(-2, 2, (1, self.width)))
        self.b_init.append(np.random.uniform(-2, 2, 1))

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return DNN.sigmoid(x)*(1-DNN.sigmoid(x))

    @staticmethod
    def loss(x, y):
        return y*np.log(x) + (1-y)*np.log(1-x)

    @staticmethod
    def dloss(x, y):
        return (y/x) - ((1-y)/(1-x))

    @staticmethod
    def euclidean_norm(W, b):
        s = 0
        L = len(W)-1
        for l in range(L+1):
            s += np.sum(W[l]**2) + np.sum(b[l]**2)
        return np.sqrt(s)

    def feedforward(x, W, b, label):
        L = len(W)-1
        h = []
        a = []
        h.append(x.reshape(-1, 1))
        for k in range(L+1):
            a.append(W[k] @ h[k] + b[k])
            h.append(DNN.sigmoid(a[k]))
        h[L+1] = DNN.sigmoid(a[L])
        y_hat = h[L+1]
        return float(DNN.loss(y_hat, label)), a, h[0:L+1], float(y_hat)

    def gradients_per_obs(x, W, b, label):
        L = len(W)-1
        nablaW = [0] * (L+1)
        nablab = [0] * (L+1)
        ff = DNN.feedforward(x, W, b, label)
        y_hat = ff[3]
        a = ff[1]
        h = ff[2]
        g = DNN.dloss(y_hat, label)
        for k in range(L, -1, -1):
            g = g * DNN.dsigmoid(a[k])
            nablab[k] = g
            nablaW[k] = g.reshape(-1, 1) @ (h[k].reshape(1, -1))
            g = W[k].T @ g.reshape(-1, 1)
        return nablaW, nablab

    @staticmethod
    def objective_function(X, labels, W, b):
        n = X.shape[0]
        obj = 0
        for i in range(n):
            obj += DNN.feedforward(X[i], W, b, labels[i])[0]
        return -obj / n

    def full_gradient(self, X, labels, W, b):
        n = X.shape[0]
        L = len(W)-1
        fullnablaW = [0] * (L+1)
        fullnablab = [0] * (L+1)
        for l in range(L+1):
            for i in range(n):
                per_obsW = DNN.gradients_per_obs(X[i], W, b, labels[i])[0]
                per_obsb = DNN.gradients_per_obs(X[i], W, b, labels[i])[1]
                fullnablaW[l] -= per_obsW[l] / n
                fullnablab[l] -= per_obsb[l] / n
        return fullnablaW, fullnablab

    def fit(self, X, labels, iterations=100, epsilon=0.01, rho1=0.9, rho2=0.999):
        L = self.L
        width = self.width
        n = X.shape[0]
        self.W = self.W_init
        self.b = self.b_init
        obj_old = DNN.objective_function(X=X, labels=labels, W=self.W, b=self.b)
        obj_new = 0
        t = 0
        stab = 10**(-8)
        sW = [0] * (L+1)
        sb = [0] * (L+1)
        rW = [0] * (L+1)
        rb = [0] * (L+1)
        accuracy = 'None'

        while abs(obj_old-obj_new) > 10**(-8):
            t += 1
            if t == iterations:
                break
            G = self.full_gradient(X, labels, self.W, self.b)
            print(f"Iteration no. {t}, objective function is {obj_new}, training accuracy is {accuracy}")
            obj_old = obj_new
            for l in range(L+1):
                sW[l] = (rho1 * sW[l] + (1 - rho1) * G[0][l])
                sb[l] = (rho1 * sb[l] + (1 - rho1) * G[1][l])
                rW[l] = (rho2 * rW[l] + (1 - rho2) * (G[0][l] * G[0][l]))
                rb[l] = (rho2 * rb[l] + (1 - rho2) * (G[1][l] * G[1][l]))

            deltaW = [None] * (L+1)
            deltab = [None] * (L+1)
            for l in range(L+1):
                deltaW[l] = -epsilon * (sW[l] / (np.sqrt(rW[l]) + stab))
                deltab[l] = -epsilon * (sb[l] / (np.sqrt(rb[l]) + stab))

            for l in range(L+1):
                self.W[l] = self.W[l] + deltaW[l]
                self.b[l] = self.b[l] + deltab[l]

            obj_new = DNN.objective_function(X=X, labels=labels, W=self.W, b=self.b)

            pr = []
            for i in range(n):
                pr.append(DNN.feedforward(X[i], self.W, self.b, labels[i])[3])

            prr = np.array(pr)
            preds = np.where(prr > 0.5, 1, 0)
            accuracy = sum(preds == labels) / n

    def evaluate(self, X_test, y_test):
        n = X_test.shape[0]
        predictions = []
        for i in range(n):
            prediction = DNN.feedforward(X_test[i], self.W, self.b, y_test[i])[3]
            predictions.append(prediction)
        predictions = np.array(predictions)
        binary_predictions = np.where(predictions > 0.5, 1, 0)
        accuracy = np.sum(binary_predictions == y_test) / n
        return accuracy