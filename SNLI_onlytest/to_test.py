from train2 import evaluate, train_main

def main():
    data_train, data_val, data_test, compute_train_costerror, compute_costerror, params = train_main()
    cost_val, error_val = evaluate(data_val, compute_costerror)
    cost_test, error_test = evaluate(data_test, compute_costerror)
    # error_train = error_train/len(data_train)
    error_val = error_val/len(data_val)
    error_test = error_test/len(data_test)
    print("error dev %f test %f" % (error_val, error_test))

if __name__ == '__main__':
    main()
