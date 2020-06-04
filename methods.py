import random
import math


def mu(x):
    n = len(x)
    x_sum = sum(x)
    return (1.0 / n) * x_sum


def sigma(x):
    n = len(x)
    mu_ = mu(x)
    sum_ = []
    for x_i in x:
        sum_.append((x_i - mu_) ** 2)
    return math.sqrt((1.0 / n) * sum(sum_))

def get_standardization_prms(x):
    m = len(x[0])
    n = len(x)
    counts_mu = []
    counts_sigma = []
    for i in range(m):
        row = [x[j][i] for j in range(n)]
        count_mu = mu(row)
        count_sigma = sigma(row)
        counts_mu.append(count_mu)
        counts_sigma.append(count_sigma)
    return counts_mu, counts_sigma


def standardization(x, counts_mu, counts_sigma):
    m = len(x[0])
    n = len(x)
    for i in range(m):
        count_mu = counts_mu[i]
        count_sigma = counts_sigma[i]
        for j in range(n):
            x[j][i] = (x[j][i] -  count_mu) / count_sigma


def get_normalization_min_max(x):
	n = len(x)
	m = len(x[0])
	counts_min = []
	counts_max = []
	for i in range(m):
		x_min = 2**100
		x_max = 0
		for j in range(n):
			if x[j][i] > x_max:
				x_max = x[j][i]
			if x[j][i] < x_min:
				x_min = x[j][i]
		counts_min.append(x_min)
		counts_max.append(x_max)
	return counts_min, counts_max


def normalization(x, counts_min, counts_max):
	n = len(x)
	m = len(x[0])
	for i in range(m):
		x_min = counts_min[i]
		x_max = counts_max[i]
		for j in range(n):
			x[j][i] = (x[j][i] - x_min) / (x_max - x_min)


def loading_t(name_file):
    result = []
    with open(name_file) as name_file_:
        for line in name_file_.readlines():
            result.append([float(x) for x in line.split(',')])
    return result


def loading_v(name_file):
    result = []
    with open(name_file) as name_file_:
        for line in name_file_.readlines():
            result.append(float(line))
    return result


def separation(x, y):
    n = len(x)
    x_training = []
    y_training = []
    x_validate = []
    y_validate = []
    x_testing = []
    y_testing = []
    for i in range(n):
        p = random.random()
        if p < 0.7:
            x_training.append(x[i])
            y_training.append(y[i])
        else:
            if p < 0.9:
                x_validate.append(x[i])
                y_validate.append(y[i])
            else:
                x_testing.append(x[i])
                y_testing.append(y[i])

    return x_training, y_training, x_validate, y_validate, x_testing, y_testing


def loss(x, y, t):
    n = len(x)
    m = len(x[0])
    n_sum = 0
    for i in range(n):
        m_sum = 0
        for j in range(m):
            m_sum = m_sum + x[i][j] * t[j]
        n_sum = n_sum + (m_sum - y[i]) ** 2
    return n_sum / n


def train(x, y, x_validate, y_validate, loss_file=None):
    prm = []
    m = len(x[0])
    for i in range(m):
        prm.append(random.random())
    step = 0
    n = len(x)
    while True:
        df = []
        eps = 0.00001
        for k in range(m):
            sum_n = 0
            for i in range(n):
                sum_m = 0
                for j in range(m):
                    sum_m = sum_m + x[i][j] * prm[j]
                sum_n = sum_n + (sum_m - y[i]) * x[i][k]
            sum_n = sum_n / n
            df.append(sum_n)
        if all(-eps < value < eps for value in df):
            break
        for i in range(m):
            prm[i] = prm[i] - df[i]
        step = step + 1
        if loss_file:
            my_loss = loss(x, y, prm)
            loss_validate = loss(x_validate, y_validate, prm)
            result = ("%result\t%result\t%result" % (step, my_loss, loss_validate)).replace(".", ",")
            print(result, file=loss_file)
    return prm


def use(x, prm):
    y = []
    for x_ in x:
        y_ = 0
        for i in range(len(x_)):
            y_ = y_ + x_[i] * prm[i]
        y.append(y_)
    return y


def c_determination(x_test, prm, y_test):
    y = []
    for x_ in x_test:
        y_ = 0
        for i in range(len(x_)):
            y_ = y_ + x_[i] * prm[i]
        y.append(y_)
    n = len(y)
    count_mu = mu(y)
    f_1 = 0
    for i in range(n):
        f_1 += (y[i] - count_mu) ** 2
    f_2 = 0
    for i in range(n):
        f_2 += (y[i] - y_test[i]) ** 2
    r_2 = 1 - f_2 / f_1
    return r_2


def loss_lg(x, y, t):
    n = len(x)
    f = []
    for x_ in x:
        z = 0
        for i in range(len(x_)):
            z = z + x_[i] * t[i]
        if z > 20:
            z = 20
        if z < -20:
            z = -20
        f_ = 1 / (1 + math.exp(-z))
        f.append(f_)
    result = 0
    for i in range(len(f)):
        e = -y[i] * math.log(f[i]) - (1 - y[i]) * math.log(1 - f[i])
        result += e
    return result / n


def train_lg(x, y, x_validate, y_validate, prm=None, file_loss=None):
    n = len(x)
    m = len(x[0])
    if prm is None:
        prm = []
        for i in range(m):
            prm.append(random.random())
    step = 0
    while step < 400:
        df = []
        eps = 0.00001
        # k - индекс оптимизируещего параметра
        for k in range(m):
            n_sum = 0
            for i in range(n):
                m_sum = 0
                for j in range(m):
                    m_sum = m_sum + x[i][j] * prm[j]
                b = bool(m_sum)
                n_sum = n_sum + (b - y[i]) * x[i][k]
            df.append(n_sum)
        if all(-eps < value < eps for value in df):
            break
        for i in range(m):
            prm[i] = prm[i] - df[i]
        step = step + 1
        if file_loss:
            my_loss = loss_lg(x, y, prm)
            loss_validate = loss_lg(x_validate, y_validate, prm)
            result = ("%result\t%result\t%result" % (step, my_loss, loss_validate)).replace(".", ",")
            print(result, end="", file=file_loss)
    return prm


def use_lg(X, t):
    y = []
    for x in X:
        f = 0
        for i in range(len(x)):
            f = f + x[i] * t[i]
        y_ = bool(f)
        y.append(y_)
    return y


def bool(z):
    if z > 200:
        z = 200
    if z < -200:
        z = -200
    a = 1 / (1 + math.exp(-z))
    if a > 0.5:
        y = 1
    else:
        y = 0
    return y


def precision(y, y_test):
    n = len(y)
    errors = 0
    for i in range(n):
        if y[i] != y_test[i]:
            errors = errors + 1
    return 1 - float(errors) / n