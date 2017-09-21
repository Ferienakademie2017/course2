import time
import utils

ll = utils.LossLogger(gui=True)
for i in range(100):
    ll.logLoss(i, 100 - i)
    time.sleep(0.1)
