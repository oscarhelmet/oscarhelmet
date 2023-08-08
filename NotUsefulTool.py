import time 
import random


print("Starting training model 'DBNet_Text_Rec_Big5', using GPU #0 - NVIDA RTX 3090 TI, VRAM Alloc: 16Gib")

# Try to print a fake training epoch 
def train():
    n = 1
    while n < 200:
        for i in range(1339):
            print(f"Epoch {n}:[" + '=' * int(i / 33) + '>' + ' ' * (40 - int(i / 33)) + f"]  {i}/1338" + f" losses:{(1.55/n)*random.uniform(0.88,1.08):.4f} - binary classification val-accracy:{100-(1.55/n)*random.uniform(0.88,1.08):.4f}", end='\r',flush=True)
            time.sleep(0.02)
        print(f"\r\nFinished {n} Epoch, the val-accuracy after this epoch: {random.uniform(98.92, 99.99):.4f}")
        n+=1

while True:
    train()
    print("Finish Training. Saving the model....")

print("I have no idea why this code is written")
# probably I wanna sleep in the office and act like the model is training.
# I am clever, haha
