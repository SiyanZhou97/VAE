import numpy as np
import multiprocessing as mp
import time
class test():
    def __init__(self):
        self.v=np.array([0,0,0,0])
        self.results=[]

    def _collect_result(self, result):
        self.results.append(result)

    def _exc(self):
        print('self.v={}'.format(self.v))
        return self.v

    def _pri(self,idx):
        print('idx:{},cpu:{}'.format(idx,mp.current_process()))
        self.v[idx-1]=idx
        time.sleep(5)
        return self._exc()

    def run(self):
        print('start')
        pool=mp.Pool(2)
        for idx in [1,2,3,4]:
            pool.apply_async(self._pri, args=(idx,), callback=self._collect_result)
        pool.close()
        pool.join()
        print('end')

t=test()
starttime = time.time()
t.run()
print('That took {} seconds'.format(time.time() - starttime))
print(t.results)
