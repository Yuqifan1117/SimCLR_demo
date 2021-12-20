import csv
from matplotlib import pyplot as plt

filename = 'results/128_0.5_200_256_500_statistics.csv'
epoch,train_loss,test_acc1,test_acc5 = [],[],[],[]
fig1 = plt.figure()
with open(filename, encoding='utf-8') as f :
    reader = csv.reader(f)
    header = next(reader)
    
    for row in reader:
        epoch.append(int(row[0]))
        train_loss.append(float(row[1]))
        test_acc1.append(float(row[2]))
        test_acc5.append(float(row[3]))


plt.plot(epoch,train_loss)
plt.title('train_loss in 500 epochs')
plt.xlabel("Epoch(s)")
plt.ylabel("Loss")
plt.savefig('results/train_loss.png')
plt.show()
print(test_acc1)
fig2 = plt.figure()
plt.plot(epoch,test_acc1,label='test_acc1')
plt.plot(epoch,test_acc5,label='test_acc5')
plt.title('test_acc in 500 epochs')
plt.xlabel("Epoch(s)")
plt.ylabel("Test Acc")
plt.savefig('results/test_acc.png')
