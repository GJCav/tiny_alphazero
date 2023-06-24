import time
from functools import lru_cache

class C:
    def __init__(self) -> None:
        self.n = 20

    @lru_cache(maxsize=3)
    def val(self, s):
        time.sleep(s)
        print(f"hello {s}")
        return self.n
    
c = C()
c.val(2)
c.val(2)
c.val(2)
c.val(1)