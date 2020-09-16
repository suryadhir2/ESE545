#ESE 545 Project 1

def create_k_shingles(text, k):
    k_shingles = set(text[i:i+k] for i in range(len(text) - k + 1))
    return k_shingles

def shingle_index(shingle):
    index = 0
    term = len(shingle) - 1
    for element in shingle:
        acii = ord(element)
        if acii == 32:
            index += 36*(37**term)
            term = term - 1
        elif acii < 58:
            index += (acii - 48)*(37**term)
            term = term - 1
        else:
            index += (acii - 87)*(37**term)
            term = term - 1
    return index

def isPrime(n):
    if n==2 or n==3: return True
    if n%2==0 or n<2: return False
    for i in range(3, int(n**0.5)+1, 2):
        if n%i==0:
            return False    
    return True

def getPrime(R):
    while True:
        if isPrime(R):
            return R
        else:
            R = R + 1
            
def hash_r(R):
    a, b = np.random.randint(1, R-1, size=2)
    R = getPrime(R)
    def inner(r):
        return (a * r + b) % R
    return inner

def vhash(r, p): 
    a = np.random.randint(1, p-1, size=r)
    b = np.random.randint(1, p-1, size=r)
    p = getPrime(p)
    def inner(v):
        return ((a * v + b) % p).sum()
    return inner

def combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return 
    indices = list(range(r))
    yield frozenset(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return 
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield frozenset(pool[i] for i in indices)
