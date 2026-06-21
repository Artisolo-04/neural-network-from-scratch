import math

def sigmoid (z) :
    return 1 / (1 + math.exp(-z))

x1 , x2 = 1 , 1
target = 1

w1 , w2 , b1 = 0.5 , -0.5 , 0
w3 , w4 , b2 = -0.5 , 0.5 , 0
w5 , w6 , b3 = 1 , -1 , 0

lr = 1
epochs = 1000

for epoch in range(epochs) :

    net_h1 = x1 * w1 + x2 * w2 + b1
    h1 = sigmoid(net_h1)

    net_h2 = x1 * w3 + x2 * w4 + b2
    h2 = sigmoid(net_h2)

    net_o1 = h1 * w5 + h2 * w6 + b3
    o1 = sigmoid(net_o1)

    loss = 1/2 * (target - o1) ** 2




    delta_o1 = - (target - o1) * o1 * (1 - o1)

    dl_dw5 = delta_o1 * h1
    dl_dw6 = delta_o1 * h2
    dl_db3 = delta_o1

    delta_h1 = delta_o1 * w5 * h1 * (1 - h1)
    delta_h2 = delta_o1 * w6 * h2 * (1 - h2)

    dl_dw1 = delta_h1 * x1
    dl_dw2 = delta_h1 * x2
    dl_db1 = delta_h1

    dl_dw3 = delta_h2 * x1
    dl_dw4 = delta_h2 * x2
    dl_db2 = delta_h2


    w1 = w1 - lr * dl_dw1
    w2 = w2 - lr * dl_dw2
    b1 = b1 - lr * dl_db1
    w3 = w3 - lr * dl_dw3
    w4 = w4 - lr * dl_dw4
    b2 = b2 - lr * dl_db2
    w5 = w5 - lr * dl_dw5
    w6 = w6 - lr * dl_dw6
    b3 = b3 - lr * dl_db3

    if epoch % 100 == 0 :
        print("--------------------------------------------------------------------- Epoch ", epoch + 1)
        print (f"Epoch {epoch} : o1 = {o1:.4f} , loss = {loss:6.4f}")

print(f"\n Final o1 = {o1:.4f} , loss = {loss:6.4f}")
