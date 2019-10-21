import instrumental
from instrumental import instrument, list_instruments
import time

print(list_instruments())

myCCS = instrument('CCS')
status = myCCS.get_status()
print(type(status))

print(status)

#myCCS.reset()
#t = time.time()
#myCCS.stop_and_clear()
#print(time.time() - t)
#data = myCCS.take_data("100 ms")
#print(time.time() - t)

#myCCS.close()
