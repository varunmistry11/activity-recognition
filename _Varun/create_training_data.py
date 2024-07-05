import prepare_balanced_data_csv as balanced
import prepare_unbalanced_data_csv as unbalanced
import prepare_balanced_raw_data_csv as balanced_raw
import prepare_unbalanced_raw_data_csv as unbalanced_raw

def createBalancedData():
    for N in range(3, 16):
        balanced.createDataCSV(N)

def createUnbalancedData():
    for N in range(3, 16):
        unbalanced.createDataCSV(N)

def createBalancedRawData():
    for N in range(3, 16):
        balanced_raw.createDataCSV(N)

def createUnbalancedRawData():
    for N in range(3, 16):
        unbalanced_raw.createDataCSV(N)

if __name__ == '__main__':
    createBalancedData()
    createUnbalancedData()
    createBalancedRawData()
    createUnbalancedRawData()
