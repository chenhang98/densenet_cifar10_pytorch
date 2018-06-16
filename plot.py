from matplotlib import pyplot as plt 
import numpy as np 
import csv

def watchLog(fn):
    reader = csv.reader(open(fn))
    iters, acc, loss = [], [], []
    val_acc, val_loss = [], []

    names = next(reader)
    for row in reader:        
        loss.append(float(row[1]))
        val_loss.append(float(row[2]))
        
        acc.append(float(row[3]))
        val_acc.append(float(row[4]))

        
    print("last acc/val_acc: ", acc[-1], val_acc[-1])  
    print("max acc/val_acc: ", max(acc), max(val_acc))

    plt.subplot(211)
    plt.plot(np.log(1 - np.array(acc)), 'b', label = "acc")
    plt.plot(np.log(1 - np.array(val_acc)), 'r', label = "val_acc")
    plt.legend()

    plt.subplot(212)
    plt.plot(np.log(loss[2:]), 'b', label = "loss")
    plt.plot(np.log(val_loss[2:]), 'r', label = "val_loss")
    plt.legend()
    plt.show()

def watch(fn):
    reader = csv.reader(open(fn))
    iters, acc, loss = [], [], []
    val_acc, val_loss = [], []

    names = next(reader)
    for row in reader:        
        loss.append(float(row[1]))
        val_loss.append(float(row[2]))
        
        acc.append(float(row[3]))
        val_acc.append(float(row[4]))

        
    print("last acc/val_acc: ", acc[-1], val_acc[-1])
    print("max acc/val_acc: ", max(acc), max(val_acc))
    print("mean of last 10: ", np.mean(acc[-10:]), np.mean(val_acc[-10:]))

    print(np.argmax(val_acc))

    plt.subplot(211)
    plt.plot(np.array(acc), 'b', label = "acc")
    plt.plot(np.array(val_acc), 'r', label = "val_acc")
    plt.legend()

    plt.subplot(212)
    plt.plot(loss[2:], 'b', label = "loss")
    plt.plot(val_loss[2:], 'r', label = "val_loss")
    plt.legend()
    plt.show()


def compare(fp1, fp2):
    reader1 = csv.reader(open(fp1))
    reader2 = csv.reader(open(fp2))
    
    iters1, acc1, loss1 = [], [], []
    iters2, acc2, loss2 = [], [], []
    
    val_acc1, val_loss1 = [], []
    val_acc2, val_loss2 = [], []

    names, _ = next(reader1), next(reader2)

    for row in reader1:        
        loss1.append(float(row[1]))
        val_loss1.append(float(row[2]))
        
        acc1.append(float(row[3]))
        val_acc1.append(float(row[4]))

    for row in reader2:        
        loss2.append(float(row[1]))
        val_loss2.append(float(row[2]))
        
        acc2.append(float(row[3]))
        val_acc2.append(float(row[4]))
        
    
    plt.subplot(211)
    plt.plot(acc1, 'b', label = "acc1") 
    plt.plot(val_acc1, 'r', label = "val_acc1")
    plt.plot(acc2, 'g', label = "acc2")
    plt.plot(val_acc2, 'y', label = "val_acc2")
    plt.legend()
    
    plt.subplot(212)
    plt.plot(loss1[2:], 'b', label = "loss1")
    plt.plot(val_loss1[2:], 'r', label = "val_loss1")
    plt.plot(loss2[2:], 'g', label = "loss2")
    plt.plot(val_loss2[2:], 'y', label = "val_loss2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    watch("bak/log.csv")