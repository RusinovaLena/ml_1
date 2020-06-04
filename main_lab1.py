import methods as m

v = 1

if(v == 1):
    print("Линейная регрессия")
    x_file = r'Линейная регрессия\train\t1_linreg_x_train.csv'
    y_file = r'Линейная регрессия\train\t1_linreg_y_train.csv'
    loss_file = open("Линейная регрессия\loss.tsv", "w")
    x = m.loading_t(x_file)
    y = m.loading_v(y_file)
    counts_mu, counts_sigma = m.get_standardization_prms(x)
    m.standardization(x, counts_mu, counts_sigma)
    counts_min, counts_max = m.get_normalization_min_max(x)
    m.normalization(x, counts_min, counts_max)
    x_train, y_train, x_validate,  y_validate, x_test, y_test = m.separation(x, y)
    prm = m.train(x_train, y_train, x_validate, y_validate, loss_file=loss_file)
    print("c_determination", m.c_determination(x_test, prm, y_test))
    x_file = r'Линейная регрессия\test\t1_linreg_x_test.csv'
    y_file = r'Линейная регрессия\test\lab1_1.csv'
    x = m.loading_t(x_file)
    counts_mu_test, counts_sigma_test = m.get_standardization_prms(x)
    m.standardization(x, counts_mu, counts_sigma)
    counts_min_test, counts_max_test = m.get_normalization_min_max(x)
    print("counts_min: ", counts_min)
    print("counts_min_test: ", counts_min_test)
    print("counts_max: ", counts_max)
    print("counts_max_test: ", counts_max_test)
    print("mu: ", counts_mu)
    print("mu_test: ", counts_mu_test)
    print("sigma: ", counts_sigma)
    print("sigma_test: ", counts_sigma_test)
    m.normalization(x, counts_min, counts_max)
    result = m.use(x, prm)
    with open(y_file, "w") as f:
        for v in result:
            print(v, file=f)

else:

    print("Логистическая регрессия")
    x_file = r'Логистическая регрессия\train\t1_logreg_x_train.csv'
    y_file = r'Логистическая регрессия\train\t1_logreg_y_train.csv'
    loss_file_lg = open("Логистическая регрессия\loss.tsv", "w")
    x = m.loading_t(x_file)
    y = m.loading_v(y_file)
    counts_mu, counts_sigma = m.get_standardization_prms(x)
    m.standardization(x, counts_mu, counts_sigma)
    counts_min, counts_max = m.get_normalization_min_max(x)
    m.normalization(x, counts_min, counts_max)
    x_train, y_train, x_validate,  y_validate, x_test, y_test = m.separation(x, y)
    prm = m.train_lg(x_train, y_train, x_validate, y_validate, file_loss=loss_file_lg)
    res = m.use_lg(x_test, prm)
    print("precision: ", m.precision(res, y_test))
    x_file = r'Логистическая регрессия\test\t1_logreg_x_test.csv'
    y_file = r'Логистическая регрессия\test\lab1_2.csv'
    x = m.loading_t(x_file)
    counts_mu_test, counts_sigma_test = m.get_standardization_prms(x)
    m.standardization(x, counts_mu, counts_sigma)
    counts_min_test, counts_max_test = m.get_normalization_min_max(x)
    print("counts_min: ", counts_min)
    print("counts_min_test: ", counts_min_test)
    print("counts_max: ", counts_max)
    print("counts_max_test: ", counts_max_test)
    print("mu: ", counts_mu)
    print("mu_test: ", counts_mu_test)
    print("sigma: ", counts_sigma)
    print("sigma_test: ", counts_sigma_test)
    m.normalization(x, counts_min, counts_max)
    result = m.use_lg(x, prm)
    with open(y_file, "w") as f:
        for v in result:
            print(v, file=f)